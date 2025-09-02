#!/usr/bin/env python3
import pygame
import time
import lcm
from mbot_lcm_msgs.twist2D_t import twist2D_t

class GamepadTeleop:
    CONTROLLER_CONFIG = {
        "deadzone": 0.05,
        "l1_button": 6, 
        "l2_button": 8,
        "r1_button": 7,
        "r2_button": 9,
    }
    INITIAL_SETTINGS = {
        "max_linear": 0.1,
        "max_angular": 0.5,
        "linear_step": 0.02,
        "angular_step": 0.1,
    }
    HARDCODED_LIMITS = {
        "max_linear": 0.2,
        "max_angular": 1.0,
    }

    def __init__(self):
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        
        # Initialize LCM
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        
        # Controller settings
        self.hardcoded_max_linear = self.HARDCODED_LIMITS['max_linear']  # meters per second
        self.hardcoded_max_angular = self.HARDCODED_LIMITS['max_angular']  # radians per second
        
        # Current maximum velocities (adjustable by buttons)
        self.max_linear_velocity = self.INITIAL_SETTINGS['max_linear']
        self.max_angular_velocity = self.INITIAL_SETTINGS['max_angular']
        
        # Speed adjustment settings
        self.linear_speed_step = self.INITIAL_SETTINGS['linear_step']  # m/s per button press
        self.angular_speed_step = self.INITIAL_SETTINGS['angular_step']  # rad/s per button press
        
        # Check for joystick
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")
            
        # Initialize the first joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"\nInitialized {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print("\nControl scheme:")
        print("  - Left stick Up/Down: Forward/Backward movement")
        print("  - Right stick Left/Right: Turning")
        print("  - Left shoulder buttons (L1/L2): Increase/decrease max linear speed")
        print("  - Right shoulder buttons (R1/R2): Increase/decrease max angular speed")
        print(f"  - Current max linear speed: {self.max_linear_velocity:.2f} m/s")
        print(f"  - Current max angular speed: {self.max_angular_velocity:.2f} rad/s")
        print("Press Ctrl+C to exit\n")

    def publish_velocity(self, vx, wz):
        """Publish velocity command to LCM"""
        command = twist2D_t()
        command.vx = vx
        command.wz = wz
        self.lcm.publish("MBOT_VEL_CMD", command.encode())
        # print(f"Published velocity command - vx: {vx:.3f} m/s, wz: {wz:.3f} rad/s")
    
    def adjust_max_linear_velocity(self, increase=True):
        """Adjust the maximum linear velocity"""
        if increase:
            self.max_linear_velocity = min(self.max_linear_velocity + self.linear_speed_step, self.hardcoded_max_linear)
        else:
            self.max_linear_velocity = max(self.max_linear_velocity - self.linear_speed_step, 0)
        print(f"Max linear velocity adjusted to: {self.max_linear_velocity:.2f} m/s")
    
    def adjust_max_angular_velocity(self, increase=True):
        """Adjust the maximum angular velocity"""
        if increase:
            self.max_angular_velocity = min(self.max_angular_velocity + self.angular_speed_step, self.hardcoded_max_angular)
        else:
            self.max_angular_velocity = max(self.max_angular_velocity - self.angular_speed_step, 0)
        print(f"Max angular velocity adjusted to: {self.max_angular_velocity:.2f} rad/s")

    def run(self):
        try:
            print("Starting control loop...")
            # For button press tracking (to avoid repeated triggers)
            button_states = [False] * self.joystick.get_numbuttons()
            
            while True:
                # Handle events
                pygame.event.pump()
                for event in pygame.event.get():
                    if event.type == pygame.JOYAXISMOTION:
                        print(f"\nRaw joystick axes values:")
                        for i in range(self.joystick.get_numaxes()):
                            print(f"Axis {i}: {self.joystick.get_axis(i):.3f}")
                
                current_states = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
                
                # L1 (increase linear speed)
                if current_states[self.CONTROLLER_CONFIG['l1_button']] and not button_states[self.CONTROLLER_CONFIG['l1_button']]:
                    self.adjust_max_linear_velocity(increase=True)
                
                # L2 (decrease linear speed)
                if current_states[self.CONTROLLER_CONFIG['l2_button']] and not button_states[self.CONTROLLER_CONFIG['l2_button']]:
                    self.adjust_max_linear_velocity(increase=False)
                
                # R1 (increase angular speed)
                if current_states[self.CONTROLLER_CONFIG['r1_button']] and not button_states[self.CONTROLLER_CONFIG['r1_button']]:
                    self.adjust_max_angular_velocity(increase=True)
                
                # R2 (decrease angular speed)
                if current_states[self.CONTROLLER_CONFIG['r2_button']] and not button_states[self.CONTROLLER_CONFIG['r2_button']]:
                    self.adjust_max_angular_velocity(increase=False)
                
                # Update button states
                button_states = current_states
                
                # Get joystick values
                raw_linear = self.joystick.get_axis(1)  # Left stick Y-axis
                raw_angular = self.joystick.get_axis(2)  # Right stick X-axis
                
                # Note: Joystick values are inverted (-1 is up/right, 1 is down/left)
                linear_vel = -raw_linear * self.max_linear_velocity
                angular_vel = -raw_angular * self.max_angular_velocity
                
                # Apply deadzone to prevent drift
                if abs(linear_vel) < self.CONTROLLER_CONFIG['deadzone']:
                    linear_vel = 0
                if abs(angular_vel) < self.CONTROLLER_CONFIG['deadzone']:
                    angular_vel = 0
                
                # Publish command
                self.publish_velocity(linear_vel, angular_vel)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nExiting...")
            # Send zero velocity command before exiting
            self.publish_velocity(0, 0)
        except Exception as e:
            print(f"\nError occurred: {e}")
        finally:
            pygame.quit()

if __name__ == "__main__":
    teleop = GamepadTeleop()
    teleop.run() 