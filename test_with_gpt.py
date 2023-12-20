
prompt_template = '''
You are trying to design the state code of a reinforcement learning algorithm for adaptive bit rate.

Adaptive Bit Rate (ABR) is a streaming technology used in multimedia applications, particularly in video streaming, to optimize the delivery of content over networks with varying bandwidth conditions. The main goal of ABR is to provide a smooth and uninterrupted viewing experience for users by dynamically adjusting the quality of the video based on the available network conditions.

Basically, we have the historic information:
    - `bit_rate_list`: The historic bit rate we chose. bit_rate_list[-1] is the latest chosen bit rate; bit_rate_list[-2] is the bit rate chosen before bit_rate_list[-1].
    - `buffer_size_list`: The historic buffer_size. buffer_size[-1] means the latest buffer size we have. If the current buffer size is low, we may have to suffer from high penalty.
    - `video_chunk_size_list`: The video chunk size in history.
    - `delay_list`: The download time of the chunks in `video_chunk_size_list`.
    - `video_chunk_remain_list`: The remaining video chunk list.

```python
import numpy as np

def get_state(
    bit_rate_list,
    buffer_size_list,
    video_chunk_size_list,
    delay_list,
    video_chunk_remain_list,
):
    look_back_window = 8
    data = np.array([
        bit_rate_list[-look_back_window:],
        buffer_size_list[-look_back_window:],
        [video_chunk_size[-(look_back_window - i)] / delay_list[-(look_back_window - i)]  for i in range(look_back_window)],
        video_chunk_remain_list[-look_back_window:]
    ])
    return data
```

Try to improve the state design for me.
'''.strip()



import numpy as np

def get_state(
    bit_rate_list,
    buffer_size_list,
    video_chunk_size_list,
    delay_list,
    video_chunk_remain_list,
):
    look_back_window = 8
    data = np.array([
        bit_rate_list[-look_back_window:],
        buffer_size_list[-look_back_window:],
        [video_chunk_size[-(look_back_window - i)] / delay_list[-(look_back_window - i)]  for i in range(look_back_window)],
        video_chunk_remain_list[-look_back_window:]
    ])
    return data



import numpy as np

def get_state(
    bit_rate_list,
    buffer_size_list,
    video_chunk_size_list,
    delay_list,
    video_chunk_remain_list,
    total_video_chunks,
):
    look_back_window = 8
    # Feature scaling for buffer size and bitrate by normalizing them to their respective maximum values
    max_buffer_size = np.max(buffer_size_list)
    max_bit_rate = np.max(bit_rate_list)
    normalized_buffer = np.array(buffer_size_list[-look_back_window:]) / (max_buffer_size if max_buffer_size > 0 else 1)
    normalized_bit_rate = np.array(bit_rate_list[-look_back_window:]) / (max_bit_rate if max_bit_rate > 0 else 1)

    # Calculation of throughput with exponential smoothing
    throughputs = np.array([video_chunk_size_list[-(look_back_window - i)] / delay_list[-(look_back_window - i)] for i in range(look_back_window)])
    smoothed_throughputs = np.array([np.mean(throughputs[max(0, i-2):i+1]) for i in range(look_back_window)])  # Use a window of 3 for smoothing

    # Calculation of percentage of video remaining
    percentage_video_remaining = np.array(video_chunk_remain_list[-look_back_window:]) / total_video_chunks

    # Combine all features into a single array
    data = np.stack([
        normalized_bit_rate,
        normalized_buffer,
        smoothed_throughputs,
        percentage_video_remaining
    ])

    return data

def adaptive_smoothing(throughputs):
    # An example of adaptive exponential smoothing,
    # where alpha is adjusted dynamically based on the variability of throughput
    alpha = 1.0 / (1.0 + np.std(throughputs))
    smoothed = [throughputs[0]]
    for i in range(1, len(throughputs)):
        smoothed_value = alpha * throughputs[i] + (1 - alpha) * smoothed[i-1]
        smoothed.append(smoothed_value)
    return np.array(smoothed)

def get_state(
    bit_rate_list,
    buffer_size_list,
    video_chunk_size_list,
    delay_list,
    video_chunk_remain_list,
    total_video_chunks,
    latency_list=None  # New parameter for latency information
):
    look_back_window = 8
    max_buffer_size = np.max(buffer_size_list)
    max_bit_rate = np.max(bit_rate_list)
    normalized_buffer = np.array(buffer_size_list[-look_back_window:]) / max_buffer_size
    normalized_bit_rate = np.array(bit_rate_list[-look_back_window:]) / max_bit_rate

    # Calculate throughput and apply adaptive smoothing
    throughputs = np.array([video_chunk_size_list[-(look_back_window - i)] / delay_list[-(look_back_window - i)] for i in range(look_back_window)])
    smoothed_throughputs = adaptive_smoothing(throughputs)

    # Calculation of percentage of video remaining
    percentage_video_remaining = np.array(video_chunk_remain_list[-look_back_window:]) / total_video_chunks

    # Incorporate latency information if available
    if latency_list is not None:
        normalized_latency = np.array(latency_list[-look_back_window:]) / np.max(latency_list)

    # Calculate the rate of change for bit rate
    bit_rate_changes = np.diff(np.array(bit_rate_list[-look_back_window-1:]))

    # Combine all features into a single array
    data = np.stack([
        normalized_bit_rate,
        normalized_buffer,
        smoothed_throughputs,
        percentage_video_remaining,
        bit_rate_changes
    ])

    # Add normalized latency to the state if available
    if latency_list is not None:
        data = np.vstack([data, normalized_latency])

    return data 


```python
import numpy as np

# Constants for normalization
MAX_BUFFER_SIZE = 100.0  # hypothetical max buffer size in seconds
MAX_BIT_RATE = 5000.0  # hypothetical max bit rate in kbps
MAX_LATENCY = 1000.0  # hypothetical max latency in ms

def adaptive_smoothing(throughputs):
    # Use the same smoothing function or consider alternatives
    alpha = 1.0 / (1.0 + np.std(throughputs))
    smoothed = [throughputs[0]]
    for i in range(1, len(throughputs)):
        smoothed_value = alpha * throughputs[i] + (1 - alpha) * smoothed[i-1]
        smoothed.append(smoothed_value)
    return np.array(smoothed)

def get_state(
    bit_rate_list,
    buffer_size_list,
    video_chunk_size_list,
    delay_list,
    video_chunk_remain_list,
    total_video_chunks,
    latency_list=None
):
    look_back_window = 8

    # Normalize buffer size and bit rate using global or predefined maxima
    normalized_buffer = np.array(buffer_size_list[-look_back_window:]) / MAX_BUFFER_SIZE
    normalized_bit_rate = np.array(bit_rate_list[-look_back_window:]) / MAX_BIT_RATE

    # Calculate throughput and apply adaptive smoothing
    throughputs = np.array([video_chunk_size_list[-(look_back_window - i)] / delay_list[-(look_back_window - i)] for i in range(look_back_window)])
    smoothed_throughputs = adaptive_smoothing(throughputs)

    # Calculation of percentage of video remaining
    percentage_video_remaining = np.array(video_chunk_remain_list[-look_back_window:]) / total_video_chunks

    # Incorporate latency information
    if latency_list:
        normalized_latency = np.array(latency_list[-look_back_window:]) / MAX_LATENCY
        latency_feature = normalized_latency
    else:
        latency_feature = np.zeros(look_back_window)  # Placeholder if latency data is unavailable

    # Calculate the weighted change for bit rate over the window
    weighted_bit_rate_changes = np.convolve(np.array(bit_rate_list[-look_back_window-1:]), np.array([0.1, 0.2, 0.3, 0.4]), 'valid')

    # Combine all features into a single array
    data = np.stack([
        normalized_bit_rate,
        normalized_buffer,
        smoothed_throughputs,
        percentage_video_remaining,
        weighted_bit_rate_changes,
        latency_feature  # Add latency feature
    ])

    return data
```