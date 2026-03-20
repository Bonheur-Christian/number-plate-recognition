from collections import deque, Counter


class TemporalConfirm:
    def __init__(self, max_history=10, confirm_threshold=3):
        self.history = deque(maxlen=max_history)
        self.confirm_threshold = confirm_threshold
        self.last_confirmed = None

    def update(self, plate_text: str):
        if not plate_text:
            return None

        self.history.append(plate_text)
        counts = Counter(self.history)
        best_plate, best_count = counts.most_common(1)[0]

        if best_count >= self.confirm_threshold:
            if best_plate != self.last_confirmed:
                self.last_confirmed = best_plate
                return best_plate

        return None