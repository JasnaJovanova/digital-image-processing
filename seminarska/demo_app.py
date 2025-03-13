import cv2

def apply_gaussian_blur(image, x, y, w, h):
    face = image[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_face

def apply_pixelated_blur(image, x, y, w, h):
    face = image[y:y+h, x:x+w]
    small = cv2.resize(face, (10, 10), interpolation=cv2.INTER_LINEAR)
    pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_face

def main():
    image = cv2.imread('input.jpg')
    if image is None:
        print("Грешка: Не може да се вчита сликата.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print("Избери тип на замаглување:")
    print("1: Gaussian Blur")
    print("2: Pixelated Blur")
    choice = input("Внеси го бројот на твојот избор: ")

    for (x, y, w, h) in faces:
        if choice == '1':
            apply_gaussian_blur(image, x, y, w, h)
        elif choice == '2':
            apply_pixelated_blur(image, x, y, w, h)
        else:
            print("Невалиден избор!")
            return

    cv2.imshow('Blurred Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    try:
        cv2.imwrite('blurred_output_image.jpg', image)
        print("Сликата успешно е зачувана.")
    except Exception as e:
        print(f"Се јави грешка при зачувување на сликата: {e}")

if __name__ == "__main__":
    main()