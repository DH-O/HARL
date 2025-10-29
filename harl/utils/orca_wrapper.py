"""
Simple ORCA-inspired Collision Avoidance for HARL

A lightweight implementation inspired by ORCA (Optimal Reciprocal Collision Avoidance)
for use with simple_spread_v2 environment in hierarchical SAC.

This implementation focuses on:
1. Agent-agent collision avoidance using velocity obstacles
2. Simple repulsion-based movement toward targets
3. Conversion to 5-dimensional force actions
"""

import numpy as np


class SimpleORCAWrapper:
    """
    Simple ORCA-inspired collision avoidance wrapper.
    
    Uses basic velocity obstacle concepts without full RVO2 implementation.
    """
    
    def __init__(self, num_agents, agent_radius, max_speed, time_step, map_size=1.0):
        """
        Initialize the simple ORCA wrapper.
        
        Args:
            num_agents: Number of agents
            agent_radius: Agent collision radius
            max_speed: Maximum agent speed
            time_step: Simulation time step
            map_size: Map size (for boundary checking)
        """
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.time_step = time_step
        self.map_size = map_size  # Map boundary
        
        # Collision avoidance parameters
        self.time_horizon = 2.0  # Seconds to look ahead
        self.relaxation = 0.5  # How much to relax velocities
        self.boundary_threshold = 0.1  # Distance from boundary to trigger avoidance
        
    def parse_observation(self, obs):
        """
        Parse observation to extract positions, landmarks, and walls.
        
        Observation: [self_vel(2), self_pos(2), landmark_rel(2*N), other_rel(2*(N-1)), comm, wall_info(4*wall_count)]
        wall_info format: [rel_x, rel_y, width, height] for each wall
        
        Returns:
            agent_pos, agent_vel, landmarks, walls (all as numpy arrays)
        """
        agent_vel = obs[0:2]
        agent_pos = obs[2:4]
        
        # Extract landmarks (relative positions)
        num_landmarks = self.num_agents
        landmark_rel_start = 4
        landmarks = []
        for i in range(num_landmarks):
            rel_pos = obs[landmark_rel_start + 2*i : landmark_rel_start + 2*(i+1)]
            landmark_abs_pos = agent_pos + rel_pos
            landmarks.append(landmark_abs_pos)
        
        # Extract wall information (at the end of observation)
        # Wall info format: [rel_x, rel_y, width, height] per wall
        walls = []
        # Parse walls from the end of observation (they come after comm + other_pos)
        # Calculate starting position: 2 (vel) + 2 (pos) + 2*N (landmarks) + 2*(N-1) (other_pos) + 2*(N-1) (comm)
        other_comm_start = landmark_rel_start + 2 * num_landmarks  # After landmarks
        other_comm_length = 2 * (self.num_agents - 1) + 2 * (self.num_agents - 1)  # other_pos + comm
        wall_start = other_comm_start + other_comm_length
        
        # Try to parse walls (they come after comm)
        obs_len = len(obs)
        remaining_elements = obs_len - wall_start
        
        # Each wall has 4 elements [rel_x, rel_y, width, height]
        if remaining_elements > 0 and remaining_elements % 4 == 0:
            num_walls = remaining_elements // 4
            for i in range(num_walls):
                wall_idx = wall_start + 4 * i
                rel_x = obs[wall_idx]
                rel_y = obs[wall_idx + 1]
                width = obs[wall_idx + 2]
                height = obs[wall_idx + 3]
                
                # Convert to absolute position and bounding box
                wall_center = agent_pos + np.array([rel_x, rel_y])
                walls.append({
                    'center': wall_center,
                    'width': width,
                    'height': height
                })
            
        return agent_pos, agent_vel, np.array(landmarks), walls
    
    def check_wall_collision(self, agent_pos, agent_radius, wall):
        """
        Check if agent will collide with a wall.
        
        Args:
            agent_pos: Agent position
            agent_radius: Agent radius
            wall: Dict with 'center', 'width', 'height'
            
        Returns:
            avoid_vel: Velocity adjustment to avoid wall, or None if no collision
        """
        # Wall bounding box in absolute coordinates
        wall_center = wall['center']
        half_width = wall['width'] / 2
        half_height = wall['height'] / 2
        
        # Check if agent is near or inside wall bounding box
        dx = abs(agent_pos[0] - wall_center[0]) - half_width - agent_radius
        dy = abs(agent_pos[1] - wall_center[1]) - half_height - agent_radius
        
        # If both dx and dy are negative, agent is inside/extremely close to wall
        if dx < 0 and dy < 0:
            # Agent is very close, push away from wall center
            direction = agent_pos - wall_center
            dist = np.linalg.norm(direction)
            if dist > 1e-8:
                direction = direction / dist
                repel_strength = min(abs(dx) + abs(dy), agent_radius) / agent_radius * self.max_speed * 0.4
                return -direction * repel_strength
        
        # Check proximity to wall edges
        threshold = self.agent_radius + 0.2
        
        # Calculate minimum distance to wall
        closest_x = max(wall_center[0] - half_width, min(agent_pos[0], wall_center[0] + half_width))
        closest_y = max(wall_center[1] - half_height, min(agent_pos[1], wall_center[1] + half_height))
        
        dist_to_wall = np.linalg.norm(agent_pos - np.array([closest_x, closest_y]))
        
        if dist_to_wall < threshold:
            direction = agent_pos - np.array([closest_x, closest_y])
            if np.linalg.norm(direction) > 1e-8:
                direction = direction / np.linalg.norm(direction)
                repel_strength = (1.0 - dist_to_wall / threshold) * self.max_speed * 0.3
                return direction * repel_strength
        
        return None
    
    def check_map_collision(self, agent_pos, agent_vel, agent_radius, walls=None):
        """
        Check if agent will collide with map boundary and walls.
        Uses a simple repulsion force near boundaries and obstacles.
        
        Args:
            agent_pos: Agent current position
            agent_vel: Agent current velocity  
            agent_radius: Agent radius
            walls: List of wall dicts (optional)
            
        Returns:
            avoid_vel: Velocity adjustment to avoid boundary/walls
        """
        avoid_vel = np.zeros(2)
        
        # Check boundary collisions
        dist_to_boundary_x = self.map_size - abs(agent_pos[0]) - agent_radius
        if dist_to_boundary_x < self.boundary_threshold:
            repel_strength = (1.0 - dist_to_boundary_x / self.boundary_threshold) * self.max_speed * 0.3
            avoid_vel[0] = -np.sign(agent_pos[0]) * repel_strength
        
        dist_to_boundary_y = self.map_size - abs(agent_pos[1]) - agent_radius
        if dist_to_boundary_y < self.boundary_threshold:
            repel_strength = (1.0 - dist_to_boundary_y / self.boundary_threshold) * self.max_speed * 0.3
            avoid_vel[1] = -np.sign(agent_pos[1]) * repel_strength
        
        # Check wall collisions
        if walls is not None:
            for wall in walls:
                wall_avoid = self.check_wall_collision(agent_pos, agent_radius, wall)
                if wall_avoid is not None:
                    avoid_vel += wall_avoid
        
        return avoid_vel
    
    def compute_collision_avoidance_velocity(self, agent_pos, agent_vel, pref_vel, other_agents_pos, other_agents_vel, walls=None):
        """
        Compute collision-free velocity using simple velocity obstacle approach.
        
        This is a simplified version that:
        1. Prefers the preferred velocity
        2. Applies collision avoidance only when necessary
        3. Considers map boundaries and walls
        
        Args:
            agent_pos: Agent position
            agent_vel: Agent current velocity
            pref_vel: Preferred velocity toward target
            other_agents_pos: List of other agent positions
            other_agents_vel: List of other agent velocities
            walls: List of wall dicts (optional)
            
        Returns:
            Collision-free velocity
        """
        
        collision_weight = 0.0
        total_avoidance = np.zeros(2)

        for other_pos in other_agents_pos:
            rel_pos = other_pos - agent_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist < 2 * self.agent_radius * 1.5:
                direction = rel_pos / (dist + 1e-8)
                avoid_speed = self.max_speed * 0.5
                avoid_vel = -direction * avoid_speed
                
                # 가중치 누적
                weight = 1.0 / (dist + 0.1)
                total_avoidance += avoid_vel * weight
                collision_weight += weight

        # 평균 회피 벡터로 결정
        if collision_weight > 0:
            avg_avoidance = total_avoidance / collision_weight
            new_vel = self.relaxation * avg_avoidance + (1 - self.relaxation) * pref_vel
        else:
            new_vel = np.copy(pref_vel)
        
        # Check boundary and wall collisions
        boundary_avoid = self.check_map_collision(agent_pos, new_vel, self.agent_radius, walls)
        if np.any(boundary_avoid != 0):
            # Blend boundary avoidance with current velocity
            new_vel = 0.7 * new_vel + 0.3 * boundary_avoid
                
        # Limit speed
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = new_vel / speed * self.max_speed
        elif speed < 0.1:  # Add small random noise if too slow
            angle = np.random.uniform(0, 2 * np.pi)
            new_vel = np.array([np.cos(angle), np.sin(angle)]) * 0.1
        
        return new_vel
    
    def velocity_to_force(self, velocity):
        """Convert velocity to proper force action format."""
        vx, vy = velocity[0], velocity[1]
        
        # simple_env.py Line 208-209에 맞춰서:
        # agent.action.u[0] = action[1] - action[2]  # left - right
        # agent.action.u[1] = action[3] - action[4]  # down - up
        
        force = np.zeros(5)
        
        # 힘의 상대적 강도 계산 (0~1 범위)
        # no_action은 사용하지 않음
        force[0] = 0.0
        
        # x 방향 힘 매핑
        if vx < 0:  # left
            force[1] = 0.0
            force[2] = min(abs(vx) / self.max_speed, 1.0)
        elif vx > 0:  # right
            force[1] = min(abs(vx) / self.max_speed, 1.0)
            force[2] = 0.0
        else:
            force[1] = 0.0
            force[2] = 0.0
        
        # y 방향 힘 매핑
        if vy < 0:  # down
            force[3] = 0.0
            force[4] = min(abs(vy) / self.max_speed, 1.0)
        elif vy > 0:  # up
            force[3] = min(abs(vy) / self.max_speed, 1.0)
            force[4] = 0.0
        else:
            force[3] = 0.0
            force[4] = 0.0
        
        return force
    
    def compute_actions(self, obs, targets):
        """
        Compute actions for all agents.
        
        Args:
            obs: Observations for all agents, shape (n_threads, n_agents, obs_dim)
            targets: Target landmark indices, shape (n_threads, n_agents, 1)
            
        Returns:
            actions: Actions for all agents, shape (n_threads, n_agents, 5)
        """
        n_threads, n_agents, _ = obs.shape
        target_indices = targets[:, :, 0].astype(int)
        
        all_actions = []
        
        for thread_id in range(n_threads):
            thread_actions = []
            
            # Parse all agent observations
            agent_positions = []
            agent_velocities = []
            landmark_positions_list = []
            walls_list = []
            
            for agent_id in range(n_agents):
                pos, vel, landmarks, walls = self.parse_observation(obs[thread_id, agent_id])
                agent_positions.append(pos)
                agent_velocities.append(vel)
                if agent_id == 0:  # Use first agent's landmarks and walls (all see same)
                    landmark_positions_list = landmarks
                    walls_list = walls
            
            # Compute preferred velocities for each agent
            for agent_id in range(n_agents):
                target_idx = target_indices[thread_id, agent_id]
                agent_pos = agent_positions[agent_id]
                
                # Get target landmark
                if target_idx < len(landmark_positions_list):
                    goal = landmark_positions_list[target_idx]
                    
                    # Compute preferred velocity toward target
                    direction = goal - agent_pos
                    dist = np.linalg.norm(direction)
                    
                    if dist > 0.01:
                        # Move toward target
                        pref_vel = (direction / dist) * min(self.max_speed, dist / self.time_step)
                    else:
                        # Reached target, slow down
                        pref_vel = agent_velocities[agent_id] * 0.5
                else:
                    # Invalid target, stay still
                    pref_vel = np.array([0.0, 0.0])
                    print(f"Invalid target, stay still: {target_idx}, agent_id: {agent_id}, agent_pos: {agent_pos}, landmark_positions_list: {landmark_positions_list}")
                
                # Apply collision avoidance
                other_positions = [agent_positions[i] for i in range(n_agents) if i != agent_id]
                other_velocities = [agent_velocities[i] for i in range(n_agents) if i != agent_id]
                
                collision_free_vel = self.compute_collision_avoidance_velocity(
                    agent_pos, agent_velocities[agent_id], pref_vel, 
                    np.array(other_positions), np.array(other_velocities),
                    walls=walls_list
                )
                
                # Convert to force action
                action = self.velocity_to_force(collision_free_vel)
                thread_actions.append(action)
            
            all_actions.append(thread_actions)
        
        return np.array(all_actions)


# For backward compatibility
ORCAWrapper = SimpleORCAWrapper
