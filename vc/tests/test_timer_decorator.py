import time
from vc.timer_decorator import timer  # adjust the import as needed
import pytest
from io import StringIO
import sys


# Define a function with a known execution time
@timer
def sleep_function():
    time.sleep(2)


def test_timer_decorator():
    # Redirect stdout to capture the print output of the timer decorator
    captured_output = StringIO()
    sys.stdout = captured_output

    # Run the decorated function
    sleep_function()

    # Reset stdout to its normal configuration
    sys.stdout = sys.__stdout__

    # Extract the elapsed time from the captured output
    output = captured_output.getvalue()
    
    # Find the index of the word 'seconds' and then access the word before it
    words = output.split()
    seconds_index = words.index("seconds")
    elapsed_time = float(words[seconds_index - 1])  # Corrected index

    # Assert that the elapsed time is reasonably close to the expected value
    # The actual elapsed time might be slightly more than 2 seconds due to function overhead
    assert 2 <= elapsed_time < 2.1  # adjust the range as needed

    # Optionally, assert that the printed function name is correct
    assert "sleep_function" in output
