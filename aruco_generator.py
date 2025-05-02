# TODO: Being able to strecth the box so that it does not have to be a square

import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np

world_name = 'asger_world_urdf.world'
aruco_size = 0.25 # m
box_size = 0.4 # m

assert aruco_size < box_size, 'The ArUco code size must be smaller than the box size'

material_text = """
material Aruco/Aruco_master
{
    technique
    {
        pass
        {
            texture_unit
            {
                texture aruco_marker_master.png
            }
        }
    }
}
"""

def parse_world_file(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()
    return tree, root

def generate_aruco_marker(id, file_path, image_size=1024):

    # Calculate the marker size
    marker_size = int(image_size * aruco_size / box_size)

    # Load the predefined dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.drawMarker(aruco_dict, id, marker_size, marker_image)

    # Create a white canvas
    canvas = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Calculate the position to center the marker
    start = (image_size - marker_size) // 2
    end = start + marker_size

    # Place the marker in the center
    canvas[start:end, start:end] = marker_image

    # Save the final image
    filename = f'aruco_marker_{id}.png'
    cv2.imwrite(os.path.join(file_path, filename), canvas)


def aruco_xml(id, position, box_size):
    # Create model
    model = ET.Element('model', name=f'aruco_code_{id}')
    
    # Add static element
    static = ET.SubElement(model, 'static')
    static.text = '1'

    # Add pose element
    pose = ET.SubElement(model, 'pose')
    if len(position) == 6:
        pose.text = f'{position[0]} {position[1]} {position[2]} {position[3]} {position[4]} {position[5]}'
    else:
        pose.text = f'{position[0]} {position[1]} {position[2]} 0 0 0'
    
    # Add link element
    link = ET.SubElement(model, 'link', name=f'link_aruco_{id}')
    
    # Add collision element inside link
    collision = ET.SubElement(link, 'collision', name=f'collision_aruco_{id}')
    col_geometry = ET.SubElement(collision, 'geometry')
    col_box = ET.SubElement(col_geometry, 'box')
    col_box_size = ET.SubElement(col_box, 'size')
    col_box_size.text = f'{box_size[0]} {box_size[1]} {box_size[2]}'

    # Add visual element inside link
    visual = ET.SubElement(link, 'visual', name=f'visual_aruco_{id}')
    geometry = ET.SubElement(visual, 'geometry')
    box = ET.SubElement(geometry, 'box')
    size = ET.SubElement(box, 'size')
    size.text = f'{box_size[0]} {box_size[1]} {box_size[2]}'

    # Add material element inside visual
    material = ET.SubElement(visual, 'material')
    script = ET.SubElement(material, 'script')
    uri_scipt = ET.SubElement(script, 'uri')
    uri_scipt.text = f'model://aruco_{id}/materials/scripts'
    usi_textures = ET.SubElement(script, 'uri')
    usi_textures.text = f'model://aruco_{id}/materials/textures'
    name = ET.SubElement(script, 'name')
    name.text = f'Aruco/Aruco_{id}'

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
        world_element.append(aruco_xml(id, data['position'], data['size']))

        # Create the materials folder
        materials_path = os.path.join(os.path.dirname(__file__), 'models', f'aruco_{id}', 'materials')
        os.makedirs(os.path.join(materials_path, 'textures'), exist_ok=True)
        os.makedirs(os.path.join(materials_path, 'scripts'), exist_ok=True)

        # Generate the ArUco marker
        generate_aruco_marker(id, os.path.join(materials_path, 'textures'))

        # Generate the material file
        aruco_material_text = material_text.replace('master', str(id))

        with open(os.path.join(materials_path, 'scripts', f'aruco_{id}.material'), 'w') as f:
            f.write(aruco_material_text)
        



def save_world_file(tree, world_file_path):
    tree.write(world_file_path)

def main():
    world_file_path = os.path.join(os.path.dirname(__file__), 'worlds', world_name)
    aruco_dict = {
        0: {'position': [0, 0, box_size / 2], 'size': [box_size, box_size, box_size]},
        1: {'position': [0, 4, box_size / 2], 'size': [box_size, box_size, box_size]},
        2: {'position': [4, 0, box_size / 2], 'size': [box_size, box_size, box_size]},
        3: {'position': [4, 4, box_size / 2], 'size': [box_size, box_size, box_size]},
        #6: {'position': [2, 2, box_size / 2, 0, 0, 0.5], 'size': [box_size, box_size, box_size]},
        #7: {'position': [0.5, 3, box_size / 2, 0, 0, 1.4], 'size': [box_size, box_size, box_size]},
        #8: {'position': [2.8, 3.2, box_size / 2, 0, 0, 3], 'size': [box_size, box_size, box_size]}, 
        #9: {'position': [3.8, 2.5, box_size / 2, 0, 0, 4], 'size': [box_size, box_size, box_size]},
        #10: {'position': [2.2, 0.5, box_size / 2, 0, 0, 0.2], 'size': [box_size, box_size, box_size]},
    }

    tree, root = parse_world_file(world_file_path)
    update_aruco_models(root, aruco_dict)
    save_world_file(tree, world_file_path)

if __name__ == "__main__":
    main()