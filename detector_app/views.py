# from django.shortcuts import render
# from django.http import JsonResponse
# from .models import UploadedImage
# from tensorflow import keras
# from PIL import Image
# import numpy as np

# def upload_image(request):
#     if request.method == 'POST':
#         uploaded_file = request.FILES['image']

#         # Memuat model H5
#         model = keras.models.load_model('detector_app/model_meaty.h5')

#         # Memproses gambar dan melakukan prediksi
#         img = Image.open(uploaded_file)
#         img = img.resize((150, 150))  # Ubah ukuran gambar menjadi 150x150
#         img_array = np.array(img)
#         img_array = img_array / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         prediction = model.predict(img_array)
#         predicted_class = np.argmax(prediction)

#         # Menentukan hasil prediksi
#         if predicted_class == 0:
#             prediction_result = 'Fresh'
#         else:
#             prediction_result = 'Spoiled'

#         # Mengirimkan respons JSON
#         return JsonResponse({'result': 'success', 'prediction': prediction_result})

#     return render(request, 'upload_image.html')

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UploadedImage
from .serializers import UploadedImageSerializer
from tensorflow import keras
from PIL import Image
import numpy as np

@api_view(['POST'])
def upload_image(request):
    uploaded_file = request.FILES['image']

    # Memuat model H5
    model = keras.models.load_model('detector_app/model_meaty.h5')

    # Memproses gambar dan melakukan prediksi
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))  # Ubah ukuran gambar menjadi 150x150
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Menentukan hasil prediksi
    if predicted_class == 0:
        prediction_result = 'Fresh'
    else:
        prediction_result = 'Spoiled'

    # Simpan uploaded_file ke database
    uploaded_image = UploadedImage(image=uploaded_file)
    uploaded_image.save()

    # Serialize uploaded_image
    serializer = UploadedImageSerializer(uploaded_image)

    # Mengirimkan respons JSON
    return Response({'result': 'success', 'prediction': prediction_result, 'uploaded_image': serializer.data})
