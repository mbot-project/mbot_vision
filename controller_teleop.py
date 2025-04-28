#!/usr/bin/env python3
import pygame
import time
import lcm
from mbot_lcm_msgs.twist2D_t import twist2D_t

class GamepadTeleop:
    def __init__(self):
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        
        # Initialize LCM
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        
        # Controller settings
        self.max_linear_velocity = 0.3  # meters per second
        self.max_angular_velocity = 1.5  # radians per second
        
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
        print("Press Ctrl+C to exit\n")

    def publish_velocity(self, vx, wz):
        """Publish velocity command to LCM"""
        command = twist2D_t()
        command.vx = vx
        command.wz = wz
        self.lcm.publish("MBOT_VEL_CMD", command.encode())
        # print(f"Published velocity command - vx: {vx:.3f} m/s, wz: {wz:.3f} rad/s")

    def run(self):
        try:
            print("Starting control loop...")
            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.JOYAXISMOTION:
                        print(f"\nRaw joystick axes values:")
                        for i in range(self.joystick.get_numaxes()):
                            print(f"Axis {i}: {self.joystick.get_axis(i):.3f}")
                
                # Get joystick values
                # Left stick Y-axis (axis 1) for linear velocity
                # Right stick X-axis (axis 2) for angular velocity
                raw_linear = self.joystick.get_axis(1)  # Left stick Y-axis
                raw_angular = self.joystick.get_axis(2)  # Right stick X-axis
                
                # Note: Joystick values are inverted (-1 is up/right, 1 is down/left)
                linear_vel = -raw_linear * self.max_linear_velocity
                angular_vel = -raw_angular * self.max_angular_velocity
                
                # Apply deadzone to prevent drift
                if abs(linear_vel) < 0.05:
                    linear_vel = 0
                if abs(angular_vel) < 0.05:
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