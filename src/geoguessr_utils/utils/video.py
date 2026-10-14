def frame_index_to_time(index, fps):
    total_seconds = index / fps
    hours = int(total_seconds // 3600)
    minute_seconds = total_seconds - (hours * 3600)
    minutes = int(minute_seconds // 60)
    seconds = int(minute_seconds - (minutes * 60))
    return f"{hours:02}:{minutes:02}:{seconds:02}"
