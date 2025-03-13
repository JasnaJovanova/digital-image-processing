import glob
import os

import cv2
import np

def load_images(dir: str) -> list[np.ndarray]:
    files = glob.glob(f'{dir}/*')
    sliki = []

    for file in files:
        sliki.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

    return sliki

def sift(slika: np.ndarray) -> tuple:
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(slika, None)

def get_matches(desc_1, desc_2) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_1, desc_2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good

def main() -> None:
    posteri = load_images('posteri')
    descriptors = [sift(image) for image in posteri]

    img = input('Search image: ')
    image = cv2.imread(os.path.join('sliki', img), cv2.IMREAD_GRAYSCALE)
    kp, _ = sift(slika)

    image_descriptors = [get_matches(desc[1], slika) for desc in descriptors]
    best = image_descriptors.index(max(image_descriptors, key=len))

    poster_keypoints = cv2.drawKeypoints(posteri[best], descriptors[best][0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    slika_keypoints = cv2.drawKeypoints(slika, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    slika_1 = np.concatenate((slika, posteri[best]), axis=1)
    slika_2 = np.concatenate((slika_keypoints, poster_keypoints), axis=1)

    cv2.imshow('a', slika_1)
    cv2.imshow('b', slika_2)
    cv2.imshow('c', cv2.drawMatchesKnn(slika, kp, posteri[best], descriptors[best][0], slika_descriptors[best], None, flags=2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()