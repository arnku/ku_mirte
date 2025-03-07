import xml.etree.ElementTree as ET
import cv2
import os

aruco_master_obj_text = """
mtllib aruco_master.mtl
o Cube
v 1.000000 1.000000 -1.000000
v 1.000000 -1.000000 -1.000000
v 1.000000 1.000000 1.000000
v 1.000000 -1.000000 1.000000
v -1.000000 1.000000 -1.000000
v -1.000000 -1.000000 -1.000000
v -1.000000 1.000000 1.000000
v -1.000000 -1.000000 1.000000
vn -0.0000 1.0000 -0.0000
vn -0.0000 -0.0000 1.0000
vn -1.0000 -0.0000 -0.0000
vn -0.0000 -1.0000 -0.0000
vn 1.0000 -0.0000 -0.0000
vn -0.0000 -0.0000 -1.0000
vt 0.625000 0.500000
vt 0.875000 0.500000
vt 0.875000 0.750000
vt 0.625000 0.750000
vt 1.000000 0.000000
vt 1.000000 1.000000
vt 0.000000 1.000000
vt 0.000000 0.000000
vt 0.125000 0.500000
vt 0.375000 0.500000
vt 0.375000 0.750000
vt 0.125000 0.750000
s 0
usemtl Material
f 1/1/1 5/2/1 7/3/1 3/4/1
f 4/5/2 3/6/2 7/7/2 8/8/2
f 8/8/3 7/7/3 5/6/3 6/5/3
f 6/9/4 2/10/4 4/11/4 8/12/4
f 2/5/5 1/6/5 3/7/5 4/8/5
f 6/8/6 5/7/6 1/6/6 2/5/6
"""

aruco_master_mtl_text = """
newmtl Material
Ns 250.000000
Ka 1.000000 1.000000 1.000000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd aruco_marker_master.png
"""

def parse_world_file(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()
    return tree, root

def generate_aruco_marker(id, file_path, size=1024):
    # Load the predefined dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    marker_image = cv2.aruco.drawMarker(aruco_dict, id, size)

    # Save the marker image
    filename = f'aruco_marker_{id}.png'
    cv2.imwrite(os.path.join(file_path, filename), marker_image)

def aruco_xml(id, position, rotation):
    # Create model
    model = ET.Element('model', name=f'aruco_code_{id}')
    
    # Add static element
    static = ET.SubElement(model, 'static')
    static.text = '1'
    
    # Add pose element
    pose = ET.SubElement(model, 'pose')
    pose.text = f'{position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]}'
    
    # Add link element
    link = ET.SubElement(model, 'link', name=f'link_aruco_{id}')
    
    # Add collision element inside link
    collision = ET.SubElement(link, 'collision', name=f'collision_aruco_{id}')
    geometry = ET.SubElement(collision, 'geometry')
    mesh = ET.SubElement(geometry, 'mesh')
    scale = ET.SubElement(mesh, 'scale')
    scale.text = '0.1 0.1 0.1'  # This is the scale of the mesh
    uri = ET.SubElement(mesh, 'uri')
    uri.text = f'model://aruco_{id}/meshes/aruco_{id}.obj'
    
    # Add visual element inside link
    visual = ET.SubElement(link, 'visual', name=f'visual_aruco_{id}')
    geometry = ET.SubElement(visual, 'geometry')
    mesh = ET.SubElement(geometry, 'mesh')
    
    # Optional: pose for the visual mesh
    pose = ET.SubElement(mesh, 'pose')
    pose.text = '0 0 0.1 0 0 0'  # Adjust the pose as needed
    
    # Scale for the visual mesh
    scale = ET.SubElement(mesh, 'scale')
    scale.text = '0.1 0.1 0.1'
    
    # URI for the visual mesh
    uri = ET.SubElement(mesh, 'uri')
    uri.text = f'model://aruco_{id}/meshes/aruco_{id}.obj'

    return model

def update_aruco_models(root, aruco_dict):

    # Find the <sdf><world> element
    world_element = root.find('world')

    # Delete all ArUco models
    for model in world_element.findall('model'):
        if model.get('name').startswith('aruco_'):
            world_element.remove(model)
    
    # Add ArUco models
    for id, data in aruco_dict.items():
        # Append the model to the world element
        world_element.append(aruco_xml(id, data['position'], data['rotation']))

        os.makedirs(f'/home/arn/ros_ws/src/ku_mirte/models/aruco_{id}/meshes', exist_ok=True)
        
        generate_aruco_marker(id, f'/home/arn/ros_ws/src/ku_mirte/models/aruco_{id}/meshes')

        # Generate the obj and mtl files
        aruco_obj_text = aruco_master_obj_text.replace('aruco_master', f'aruco_{id}')
        aruco_mtl_text = aruco_master_mtl_text.replace('aruco_marker_master.png', f'aruco_marker_{id}.png')

        with open(f'/home/arn/ros_ws/src/ku_mirte/models/aruco_{id}/meshes/aruco_{id}.obj', 'w') as f:
            f.write(aruco_obj_text)
        with open(f'/home/arn/ros_ws/src/ku_mirte/models/aruco_{id}/meshes/aruco_{id}.mtl', 'w') as f:
            f.write(aruco_mtl_text)


def save_world_file(tree, world_file_path):
    tree.write(world_file_path)

def main():
    world_file_path = '/home/arn/ros_ws/src/ku_mirte/worlds/asger_world_test.world'
    aruco_dict = {
        0: {'position': [1, 1, 0.1], 'rotation': [1.5708, 0, 0]},
        1: {'position': [2, 2, 0.1], 'rotation': [1.5708, 0, 0]},
        # Add more ArUco codes and positions as needed
    }

    tree, root = parse_world_file(world_file_path)
    update_aruco_models(root, aruco_dict)
    save_world_file(tree, world_file_path)

if __name__ == "__main__":
    main()
