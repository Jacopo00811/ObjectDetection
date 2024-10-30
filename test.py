import cv2 as cv
import os 
from read_XML import read_images_and_xml


# Find the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Append the current folder path
    folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images')
    images, annotations = read_images_and_xml(folder_path)

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow("edges", edges)
    cv.imshow("edgeboxes", im)
    cv.waitKey(0)
    cv.destroyAllWindows()