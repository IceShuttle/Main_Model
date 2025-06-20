import xml.etree.ElementTree as ET
import csv
import sys

def extract_waypoints(kml_file, output_csv):
    # Define the KML namespace
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Parse the KML file
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing KML file: {e}")
        return
    
    # List to store waypoints
    waypoints = []
    
    # Find all Placemarks in the KML file
    for placemark in root.findall('.//kml:Placemark', namespace):
        name = placemark.find('kml:name', namespace)
        point = placemark.find('kml:Point', namespace)
        
        if point is not None:
            coordinates = point.find('kml:coordinates', namespace)
            if coordinates is not None and coordinates.text:
                # Split coordinates into longitude, latitude, altitude
                coords = coordinates.text.strip().split(',')
                if len(coords) >= 2:
                    try:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        alt = float(coords[2]) if len(coords) > 2 else 0.0
                        waypoint_name = name.text if name is not None else "Unnamed"
                        waypoints.append({
                            'name': waypoint_name,
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': alt
                        })
                    except ValueError as e:
                        print(f"Error parsing coordinates for placemark {name.text if name else 'Unnamed'}: {e}")
                        continue
    
    # Write waypoints to CSV
    if waypoints:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['name', 'latitude', 'longitude', 'altitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for waypoint in waypoints:
                writer.writerow(waypoint)
        print(f"Waypoints extracted and saved to {output_csv}")
    else:
        print("No waypoints found in the KML file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_waypoints.py <input_kml_file>")
        sys.exit(1)
    
    kml_file = sys.argv[1]
    output_csv = 'Data/test-path.csv'
    extract_waypoints(kml_file, output_csv)
