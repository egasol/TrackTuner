import json
import numpy as np

class Statistics:
    def __init__(self):
        self.annotation_stats = {}
        self.track_stats = {}
        self.false_positives = 0

    def add_annotation(self, frame, obj_id, obj_position):
        if obj_id not in self.annotation_stats:
            self.annotation_stats[obj_id] = {
                'lifespan': 0,
                'tracked': 0,
                'id_switches': 0,
                'last_track_id': None,
                'associated_track_ids': set()
            }
        self.annotation_stats[obj_id]['lifespan'] += 1

    def add_track(self, frame, obj_id, track_id, track_position, obj_position):
        distance = np.linalg.norm(track_position - obj_position)

        if distance <= 4:
            if track_id not in self.track_stats:
                self.track_stats[track_id] = {
                    'lifespan': 0,
                    'tracked': 0,
                    'id_switches': 0,
                    'last_obj_id': None,
                    'associated_obj_ids': set()
                }

            self.track_stats[track_id]['lifespan'] += 1
            self.track_stats[track_id]['associated_obj_ids'].add(obj_id)
            self.annotation_stats[obj_id]['associated_track_ids'].add(track_id)

            if self.annotation_stats[obj_id]['last_track_id'] != track_id:
                self.annotation_stats[obj_id]['id_switches'] += 1
                self.annotation_stats[obj_id]['last_track_id'] = track_id

            self.annotation_stats[obj_id]['tracked'] += 1
            self.track_stats[track_id]['tracked'] += 1
            return True
        return False

    def update_false_positives(self, track_id):
        self.false_positives += 1

    def calculate_statistics(self):
        for obj_id, stats in self.annotation_stats.items():
            tracked_percentage = (stats['tracked'] / stats['lifespan']) * 100
            stats['successfully_tracked'] = tracked_percentage >= 75
            stats['tracked_percentage'] = tracked_percentage

    def get_performance_metric(self):
        alpha = 1
        beta = 1
        gamma = 0.5

        performance_metric = 0
        for obj_id, stats in self.annotation_stats.items():
            tracked_percentage = stats['tracked_percentage']
            id_switches = stats['id_switches']
            loss = alpha * (1 - tracked_percentage) + beta * id_switches
            performance_metric += loss

        performance_metric += gamma * self.false_positives
        return performance_metric

    def print_statistics(self):
        print("Tracking Performance Statistics:")
        print("Annotations:")
        for obj_id, stats in self.annotation_stats.items():
            print(f"Object ID: {obj_id}")
            print(f"  Lifespan: {stats['lifespan']} frames")
            print(f"  Tracked: {stats['tracked']} frames ({stats['tracked_percentage']:.2f}%)")
            print(f"  ID Switches: {stats['id_switches']}")
            print(f"  Successfully Tracked: {stats['successfully_tracked']}")
            print(f"  Associated Track IDs: {sorted(stats['associated_track_ids'])}")

        print("\nTracks:")
        for track_id, stats in self.track_stats.items():
            print(f"Track ID: {track_id}")
            print(f"  Lifespan: {stats['lifespan']} frames")
            print(f"  Tracked: {stats['tracked']} frames")
            print(f"  ID Switches: {stats['id_switches']}")
            print(f"  Associated Object IDs: {sorted(stats['associated_obj_ids'])}")

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_data(annotations, tracks):
    stats = Statistics()

    for frame, annotation in annotations.items():
        for obj in annotation['tracks']:
            obj_id = obj['id']
            obj_position = np.array([obj['x'], obj['y'], obj['z']])
            stats.add_annotation(frame, obj_id, obj_position)

            associated_track_id = None
            tracked_objs = set()
            for track in tracks.get(frame, {}).get('tracks', []):
                track_id = track['id']
                track_position = np.array([track['x'], track['y'], track['z']])
                if stats.add_track(frame, obj_id, track_id, track_position, obj_position):
                    tracked_objs.add(track_id)

            for track in tracks.get(frame, {}).get('tracks', []):
                if track['id'] not in tracked_objs:
                    stats.update_false_positives(track['id'])

    stats.calculate_statistics()
    return stats

def main():
    annotations = load_json('annotations.json')
    tracks = load_json('tracked_objects.json')
    
    stats = process_data(annotations, tracks)
    stats.print_statistics()
    print("performance_metric:", stats.get_performance_metric())

if __name__ == "__main__":
    main()
