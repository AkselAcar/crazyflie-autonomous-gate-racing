import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_gate_csv import plot_gates
import config


class MotionPlanner3D():
    
    # def __init__(self, start, obstacles, bounds, grid_size, goal, gate_widths):
    def __init__(self, start_pos=config.STARTING_POINT, gate_pos_csv_path="./gate_pos_csv/gates_info.csv", verbose=False):

        self.starting_point = start_pos
        self.trajectory_setpoints = None
        self.path_waypoints, self.gate_pos, self.gate_orientations, self.gate_widths, self.gate_heights = self.construct_waypoints(gate_pos_csv_path, verbose=verbose)
        print(f"[INFO] Successfully loaded and constructed {len(self.path_waypoints)} gate waypoints from {gate_pos_csv_path}.")

        self.vel_lim = config.VEL_LIM                   # Velocity limit of the drone (m/s)
        self.acc_lim = config.ACC_LIM                   # Acceleration limit of the drone (m/s²)
        self.points_per_meter = config.PTS_PER_METER    # Number of points per meter 

        # Determine the number of segments of the path
        self.distances = [np.linalg.norm(np.array(self.path_waypoints[i+1]) - np.array(self.path_waypoints[i])) for i in range(len(self.path_waypoints)-1)]
        total_dist = sum(self.distances)

        target_avg_speed = config.TARGET_AVG_SPEED
        t_f = total_dist / target_avg_speed
        t_f = max(t_f, config.MIN_T_F)
        print(f"[INFO] Target average speed: {target_avg_speed} m/s")
        print(f"[INFO] Total time expected: {t_f:.1f} s")

        # Allocate time proportionally
        self.times = [0.0]
        current_time = 0.0
        for d in self.distances:
            time_fraction = (d / total_dist) if total_dist > 0 else (1.0 / len(self.distances))
            current_time += t_f * time_fraction
            self.times.append(current_time)
    
        self.times = np.array(self.times)

        self.run_planner()

    def construct_waypoints(self, gate_pos_csv_path, verbose=False):
        offset_dist = config.ADD_POINT_DIST_FROM_GATE

        gate_pos_df = pd.read_csv(gate_pos_csv_path)
        gate_pos = gate_pos_df[["x", "y", "z"]].values.T
        gate_orientations = gate_pos_df["theta"].values - np.pi/2
        gate_widths = gate_pos_df["width"].values
        gate_heights = gate_pos_df["height"].values

        if verbose:
            print(f"Loaded gate positions (shape {gate_pos.shape}):\n{gate_pos}")
            print(f"Loaded gate orientations (shape {gate_orientations.shape}):\n{gate_orientations}")
            print(f"Loaded gate widths (shape {gate_widths.shape}):\n{gate_widths}")
            print(f"Loaded gate heights (shape {gate_heights.shape}):\n{gate_heights}")

        waypoints = [self.starting_point] # Starting point (x, y, z)
        
        num_gates = gate_pos.shape[1]
        for index in range(num_gates*2):
            i = index % num_gates

            yaw = gate_orientations[i]
            p_center = gate_pos[:, i].flatten()
            
            # Approach direction from the gate orientation
            dx = offset_dist * np.cos(yaw)
            dy = offset_dist * np.sin(yaw)

            # Approach Point
            p_approach = p_center.copy()
            p_approach[0] -= dx
            p_approach[1] -= dy
            waypoints.append(tuple(p_approach))
            
            # Center Point
            waypoints.append(tuple(p_center))
            
            # Exit Point (exit opposite to approach)
            p_exit = p_center.copy()
            p_exit[0] += dx
            p_exit[1] += dy
            waypoints.append(tuple(p_exit))

            if index == num_gates*2 - 1:
                p_final = p_exit.copy()
                p_final[0] = p_exit[0] + config.FINISH_POINT_DIST * np.cos(yaw)
                p_final[1] = p_exit[1] + config.FINISH_POINT_DIST * np.sin(yaw)
                waypoints.append(tuple(p_final))
            
        return np.array(waypoints), gate_pos, gate_orientations, gate_widths, gate_heights

    def run_planner(self):    
        poly_coeffs = self.compute_poly_coefficients()
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs)

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1], #pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], #vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0], #acc  
            [60*(t**2), 24*t, 6, 0, 0, 0], #jerk
            [120*t, 24, 0, 0, 0, 0] #snap
        ])

        return A_m

    def compute_poly_coefficients(self):
        
        # Computes a minimum jerk trajectory given time and position waypoints.
        # Outputs:
        # - poly_coeffs: The polynomial coefficients for each segment of the path [6(m-1) x 3]

        # Use the following variables and the class function self.compute_poly_matrix(t) to solve for the polynomial coefficients
        
        seg_times = np.diff(self.times) #The time taken to complete each path segment
        m = len(self.path_waypoints) #Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in self.path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs):

        time_setpoints_list = []
        for i in range(len(self.times) - 1):
            # Calculate number of points proportional to this specific segment's distance
            num_points = max(2, int(self.points_per_meter * self.distances[i]))
            
            # To avoid duplicate points exactly at the waypoints where segments connect, 
            # we omit the endpoint for all segments except the very last one.
            include_endpoint = (i == len(self.times) - 2)
            seg_times = np.linspace(self.times[i], self.times[i+1], num_points, endpoint=include_endpoint)
            time_setpoints_list.extend(seg_times)
            
        time_setpoints = np.array(time_setpoints_list)
        total_points = len(time_setpoints)

        # --- 2. UPDATED ARRAY PRE-ALLOCATION ---
        x_vals, y_vals, z_vals = np.zeros((total_points,1)), np.zeros((total_points,1)), np.zeros((total_points,1))
        v_x_vals, v_y_vals, v_z_vals = np.zeros((total_points,1)), np.zeros((total_points,1)), np.zeros((total_points,1))
        a_x_vals, a_y_vals, a_z_vals = np.zeros((total_points,1)), np.zeros((total_points,1)), np.zeros((total_points,1))
        yaw_vals = np.zeros((total_points,1))

        # Extract the x,y and z direction polynomial coefficient vectors
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        for i,t in enumerate(time_setpoints):
            seg_idx = min(max(np.searchsorted(self.times, t)-1,0), len(coeff_x) - 1)
            # Determine the x,y and z position reference points at every refernce time
            x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_x[seg_idx*6:(seg_idx+1)*6])
            y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_y[seg_idx*6:(seg_idx+1)*6])
            z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z velocities at every reference time
            v_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_x[seg_idx*6:(seg_idx+1)*6])
            v_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_y[seg_idx*6:(seg_idx+1)*6])
            v_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z accelerations at every reference time
            a_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_x[seg_idx*6:(seg_idx+1)*6])
            a_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_y[seg_idx*6:(seg_idx+1)*6])
            a_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_z[seg_idx*6:(seg_idx+1)*6])

        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))

        # self.plot(obs, path_waypoints, trajectory_setpoints)
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))

        # print("Maximum flight speed: " + str(vel_max))
        # print("Average flight speed: " + str(vel_mean))
        # print("Average flight acceleration: " + str(acc_mean))
        # print("Maximum flight acceleration: " + str(acc_max))
        
        # Check that it is less than an upper limit velocity v_lim
        assert vel_max <= self.vel_lim, "The drone velocity exceeds the limit velocity : " + str(vel_max) + " m/s"
        assert acc_max <= self.acc_lim, "The drone acceleration exceeds the limit acceleration : " + str(acc_max) + " m/s²"

        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints
    
    def plot(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory setpoints
        ax.scatter(self.path_waypoints[0,0], self.path_waypoints[0,1], self.path_waypoints[0,2], c='b', marker='o', s=100, label='Starting Point')
        ax.scatter(self.path_waypoints[-1,0], self.path_waypoints[-1,1], self.path_waypoints[-1,2], c='r', marker='o', s=100, label='Finishing Point')
        ax.scatter(self.path_waypoints[1:-2,0], self.path_waypoints[1:-2,1], self.path_waypoints[1:-2,2], c='g', marker='x', s=50, label='Waypoints')
        ax.plot(self.trajectory_setpoints[:,0], self.trajectory_setpoints[:,1], self.trajectory_setpoints[:,2], c='b', label='Trajectory')
        
        # Plot gates as vertical rectangles
        for i in range(self.gate_pos.shape[1]):
            center = self.gate_pos[:, i]
            yaw = self.gate_orientations[i]
            width = self.gate_widths[i]
            height = self.gate_heights[i]
            
            # Yaw angle defines the approach direction: (cos(yaw), sin(yaw))
            # Gate left-right axis (perpendicular to approach): (-sin(yaw), cos(yaw), 0)
            # Gate up-down axis (vertical): (0, 0, 1)
            left_right = np.array([-np.sin(yaw), np.cos(yaw), 0])
            up_down = np.array([0, 0, 1])
            
            # Corner offsets from center
            half_w = width / 2
            half_h = height / 2
            
            # Four corners of the gate rectangle
            corners = [
                center - half_w * left_right - half_h * up_down,  # bottom-left
                center + half_w * left_right - half_h * up_down,  # bottom-right
                center + half_w * left_right + half_h * up_down,  # top-right
                center - half_w * left_right + half_h * up_down,  # top-left
                center - half_w * left_right - half_h * up_down,  # close the square
            ]
            corners = np.array(corners)
            
            # Plot gate rectangle
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'r-', linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xbound(-2, 2)
        ax.set_ybound(-2, 2)
        ax.set_zbound(0, 2)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    csv_file_path = "racing/gate_pos_csv/gates_info_measured.csv"
    test = MotionPlanner3D(gate_pos_csv_path=csv_file_path, verbose=False)
    gates_df = pd.read_csv(csv_file_path)
    plot_gates(gates_df, show_plot=True)
    test.plot()