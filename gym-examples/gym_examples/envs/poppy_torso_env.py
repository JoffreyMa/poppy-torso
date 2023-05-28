import gym
from gym import spaces
from pypot import vrep
from pypot.creatures import PoppyTorso
from pypot.primitive.move import Move, MovePlayer
import numpy as np
import torch
from ..utils.skeleton import *
from ..utils.quaternion import *
from ..utils.blazepose import blazepose_skeletons

class PoppyTorsoEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, render_mode=None):
        print("Initialiazing PoppyTorsoEnv")
        super(PoppyTorsoEnv, self).__init__()

        # Init various variables
        self.fps = 1
        #self.skeletons = blazepose_skeletons('/home/joffreyma/Projets/mastere_ia/cours/IA705 - Apprentissage pour la robotique/poppy-torso/mai1.mov')
        #torch.save(f = "/home/joffreyma/Projets/mastere_ia/cours/IA705 - Apprentissage pour la robotique/poppy-torso/mai1.pt", obj=self.skeletons)
        self.skeletons = torch.load(f = "/home/joffreyma/Projets/mastere_ia/cours/IA705 - Apprentissage pour la robotique/poppy-torso/mai1.pt")
        # Skeleton measurements ?
        self.poppy_lengths = torch.Tensor([
            0.0,
            0.07,
            0.18,
            0.19,
            0.07,
            0.18,
            0.19,
            0.12,
            0.08,
            0.07,
            0.05,
            0.1, 
            0.15,
            0.13,
            0.1,
            0.15,
            0.13
        ])
        # Alpha to rotate the skeleton
        self.alpha = np.pi/4.
        # Skeleton topology
        self.topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        # step counter
        self.step_counter = 0
        # epsilon, small value useful for reward computation
        self.epsilon = 1e-5

        # Normalize and change reference
        self.rota_skeletons_B = self.change_frame(self.skeletons, 'general', self.alpha, self.topology)  # I pass self and things contained in self, a bit ugly, but I am trying to understand
        self.targets, self.all_positions = self.targets_from_skeleton(self.skeletons, self.topology)

        # Set observation space
        # Observations are dictionaries with the agent's and the target's location.
        # Agent = robot, target = human
        # 2 hands + 3 dimensions
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-10, 10, shape=(2,3,), dtype=np.float32),
                "target": spaces.Box(-10, 10, shape=(2,3,), dtype=np.float32),
            }
        )
        # Set action space
        # Let's start with all the possible motors, 13 ones
        self.action_space = spaces.Box(low=-100, high=100, shape=(13,), dtype=np.float32)

        # Set the robot simulation through VREP (CoppeliaSim)
        vrep.close_all_connections()
        self.poppy = PoppyTorso(simulator='vrep')
        # Reset the robot 
        self.reset()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def step(self, action):
        #print("Taking a step in the PoppyTorsoEnv")

        # State the movement to do based on action
        for m,a in zip(self.poppy.motors, action):
            # Send the motor commands to the robot
            m.goto_position(a, 1, wait=True)

        '''
        # State the movement to do based on action
        move = Move(freq=self.fps) # could be great if possible to collect the moves later on
        new_positions = {m.name:[a, 0.] for m,a in zip(self.poppy.motors, action)}
        move.add_position(new_positions, self.step_counter)

        # Send the motor commands to the robot
        mp = MovePlayer(self.poppy, move, play_speed=1)
        mp.start()
        mp.wait_to_stop() # if I don't wait another action starts before the end of the first one
        print("joint positions of the move ",(move._timed_positions))
        '''

        # Get observations, that is to say the cartesian position of the joints
        self._agent_location = np.stack([self.poppy.l_arm_chain.position, self.poppy.r_arm_chain.position])

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = terminated or self.step_counter+1==len(self.skeletons)
        if terminated:
            print("OUI CA FINIT UN JOUR")

        # reward depending on how far the target is
        distance = torch.dist(torch.from_numpy(self._agent_location)[0], self._target_location[0])
        reward = 1/(distance+self.epsilon)

        observation = self._get_obs()
        self.step_counter+=1

        # Update target
        self._target_location = self.targets[self.step_counter+1,...] # using the step counter to determine next target

        return observation, reward, terminated, {"distance":distance}

    # Reset the position of Poppy robot
    def reset(self):
        joint_pos = { 'l_elbow_y':0.0,
                    'head_y': 0.0,
                    'r_arm_z': 0.0, 
                    'head_z': 0.0,
                    'r_shoulder_x': 0.0, 
                    'r_shoulder_y': 0.0,
                    'r_elbow_y': 0.0, 
                    'l_arm_z': 0.0,
                    'abs_z': 0.0,
                    'bust_y': 0.0, 
                    'bust_x':0.0,
                    'l_shoulder_x': 0.0,
                    'l_shoulder_y': 0.0
                    }
        for m in self.poppy.motors:
            m.goto_position(joint_pos[m.name],5)

        self._agent_location = np.stack([self.poppy.l_arm_chain.position, self.poppy.r_arm_chain.position])
        self._target_location = self.targets[0,...]
        return self._get_obs()




    # Methods to go from raw skeleton to something usable with Poppy

    def change_frame(self, skeletons, frame_name, alpha, topology):
        rota_skeletons_A = skeletons.clone()
        rota_skeletons_A[:, :, 2] = -skeletons[:, :, 1]
        rota_skeletons_A[:, :, 1] = skeletons[:, :, 2]
        center_A = rota_skeletons_A[:, 0,:].unsqueeze(1).repeat(1, len(topology), 1)
        rota_skeletons_A = rota_skeletons_A - center_A

        batch_size, n_joints, _ = rota_skeletons_A.shape
            

        # Measure skeleton bone lengths
        lengths = torch.Tensor(batch_size, n_joints)
        for child, parent in enumerate(topology):
                lengths[:, child] = torch.sqrt(
                    torch.sum(
                        (rota_skeletons_A[:, child] - rota_skeletons_A[:, parent])**2,
                        axis=-1
                    )
                )

        # Find the corresponding angles
        offsets = torch.zeros(batch_size, n_joints, 3)
        offsets[:, :, -1] = lengths
        quaternions = find_quaternions(topology, offsets, rota_skeletons_A)
            
        # Rotate of alpha
        #define the rotation by its quaternion 
        rotation = torch.Tensor([np.cos(alpha/2),  np.sin(alpha/2),0,0]).unsqueeze(0).repeat(batch_size*n_joints, 1)
        quaternions = quaternions.reshape(batch_size*n_joints, 4)
        quaternions = batch_quat_left_multiply(
                batch_quat_inverse(rotation),
                quaternions
            )
        quaternions = quaternions.reshape(batch_size, n_joints, 4)

        # Use these quaternions in the forward kinematics with the Poppy skeleton
        skeleton = forward_kinematics(
                topology,
                torch.zeros(batch_size, 3),
                offsets,
                quaternions
            )[0]
            
        outputs= skeleton.clone()
            
        return outputs


    # Determine targets from skeleton
    def targets_from_skeleton(self, source_positions, topology):
        # Works in batched
        batch_size, n_joints, _ = source_positions.shape
        
        # Measure skeleton bone lengths
        source_lengths = torch.Tensor(batch_size, n_joints)
        for child, parent in enumerate(topology):
            source_lengths[:, child] = torch.sqrt(
                torch.sum(
                    (source_positions[:, child] - source_positions[:, parent])**2,
                    axis=-1
                )
            )
        
        # Find the corresponding angles
        source_offsets = torch.zeros(batch_size, n_joints, 3)
        source_offsets[:, :, -1] = source_lengths
        quaternions = find_quaternions(topology, source_offsets, source_positions)
        
        # Re-orient according to the pelvis->chest orientation
        base_orientation = quaternions[:, 7:8].repeat(1, n_joints, 1).reshape(batch_size*n_joints, 4)
        base_orientation += 1e-3 * torch.randn_like(base_orientation)
        quaternions = quaternions.reshape(batch_size*n_joints, 4)
        quaternions = batch_quat_left_multiply(
            batch_quat_inverse(base_orientation),
            quaternions
        )
        quaternions = quaternions.reshape(batch_size, n_joints, 4)
        
        # Use these quaternions in the forward kinematics with the Poppy skeleton
        target_offsets = torch.zeros(batch_size, n_joints, 3)
        target_offsets[:, :, -1] = self.poppy_lengths.unsqueeze(0).repeat(batch_size, 1)
        target_positions = forward_kinematics(
            topology,
            torch.zeros(batch_size, 3),
            target_offsets,
            quaternions
        )[0]

        # Measure the hip orientation
        alpha = np.arctan2(
            target_positions[0, 1, 1] - target_positions[0, 0, 1],
            target_positions[0, 1, 0] - target_positions[0, 0, 0]
        )
        
        # Rotate by alpha around z
        alpha = alpha
        rotation = torch.Tensor([np.cos(alpha/2), 0, 0, np.sin(alpha/2)]).unsqueeze(0).repeat(batch_size*n_joints, 1)
        quaternions = quaternions.reshape(batch_size*n_joints, 4)
        quaternions = batch_quat_left_multiply(
            batch_quat_inverse(rotation),
            quaternions
        )
        quaternions = quaternions.reshape(batch_size, n_joints, 4)
        
        # Use these quaternions in the forward kinematics with the Poppy skeleton
        target_positions = forward_kinematics(
            topology,
            torch.zeros(batch_size, 3),
            target_offsets,
            quaternions
        )[0]
        

        
        # Return only target positions for the end-effector of the 6 kinematic chains:
        # Chest, head, left hand, left elbow, left shoulder, right hand, right elbow
        # end_effector_indices = [8, 10, 13, 12, 11, 16, 15]
        end_effector_indices = [13, 16]
        # end_effector_indices = [13, 12, 16, 15]

        return target_positions[:, end_effector_indices], target_positions