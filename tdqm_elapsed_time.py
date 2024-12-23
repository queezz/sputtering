import time
from tqdm.notebook import tqdm
import threading


def update_progress_bar(duration):
    """
    Update a tqdm progress bar for a given duration in seconds.
    """
    with tqdm(total=duration, desc="Exposure Progress", unit="s") as pbar:
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            pbar.n = int(elapsed)  # Update the bar to elapsed time
            pbar.refresh()  # Force the bar to update
            time.sleep(0.1)  # Small sleep to reduce CPU usage
        pbar.n = duration  # Ensure bar completes at the end
        pbar.refresh()


# Simulated camera operation
def simulate_camera_capture(exposure_time):
    print("Camera started capturing...")
    time.sleep(exposure_time)  # Simulate the blocking operation
    print("Camera finished capturing.")


exposure_time = 60
progress_thread = threading.Thread(target=update_progress_bar, args=(exposure_time,))
progress_thread.start()
simulate_camera_capture(exposure_time)

progress_thread.join()
