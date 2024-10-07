import os
from .models import Rating, User, Item
from django.shortcuts import render
from django.shortcuts import get_object_or_404, redirect
from .models import User  # Pastikan User adalah model pelanggan Anda
import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.metrics import mean_absolute_error
from django.http import JsonResponse

def recommendations(request):
    # Mengambil file csv
    csv_path = r'E:\Duniawi\Done\Colaborative Filtering\cf-recom\collaborative_filtering.csv'
    
    # Baca file CSV
    df = pd.read_csv(csv_path, delimiter=';', usecols=['user_id', 'item_id', 'rating'])
    
    data = Dataset.load_from_df(df, Reader())
    trainset = data.build_full_trainset()
    
    # Model SVD
    model = SVD()
    model.fit(trainset)
    
    # Semua user
    all_users = df.user_id.unique()
    
    # Semua item
    all_item = df.item_id.unique()

    # List untuk menyimpan hasil rekomendasi untuk setiap user
    all_recommendations = []

    # List untuk menyimpan nilai prediksi dan rating aktual
    all_actual_ratings = []
    all_predicted_ratings = []

    for user_id in all_users:
        # Item yang sudah di-rating oleh user
        rated_items = df[df.user_id == user_id]
        
        # Prediksi untuk item yang sudah di-rating
        for _, row in rated_items.iterrows():
            item_id = row['item_id']
            actual_rating = row['rating']
            predicted_rating = model.predict(user_id, item_id).est
            
            # Simpan rating aktual dan prediksi
            all_actual_ratings.append(actual_rating)
            all_predicted_ratings.append(predicted_rating)

        # Item yang belum di-rating oleh user
        rated = rated_items.item_id
        not_rated = [item_id for item_id in all_item if int(item_id) not in rated.values]
        
        # Prediksi rating untuk item yang belum di-rating
        score = [model.predict(user_id, item_id).est for item_id in not_rated]
        
        # Gabungkan hasil prediksi dalam bentuk dataframe
        result = pd.DataFrame({"user_id": user_id, "item": not_rated, "pred_score": score})
        result.sort_values("pred_score", ascending=False, inplace=True)

        # Simpan hasil rekomendasi untuk user ini
        all_recommendations.append(result)

    # Menghitung MAE
    mae = mean_absolute_error(all_actual_ratings, all_predicted_ratings)

    # Menggabungkan semua rekomendasi
    all_recommendations_df = pd.concat(all_recommendations)

    # Mengirim hasil ke template HTML
    context = {
        "all_recommendations": all_recommendations_df.to_dict(orient='records'),
        "mae": mae
    }
    
    return render(request, 'dashboard/dashboard.html', context)

def home(request):
    return render(request, 'dashboard/home.html')

def login(request):
    return render(request, 'login.html')

def dashboard(request):
    return render(request, 'admin.html')

def keloladata(request):
    # Mengambil semua item untuk dijadikan header kolom
    items = Item.objects.all()

    # Mengambil semua user untuk dropdown pelanggan
    pelanggan_list = User.objects.all()

    # Mengambil semua user untuk ditampilkan dalam tabel rating
    users = User.objects.all()

    # Membuat dictionary untuk menyimpan rating tiap user dan item
    user_ratings = []
    for user in users:
        user_rating = {'user': user, 'ratings': []}
        for item in items:
            rating = Rating.objects.filter(user=user, item=item).first()  # Cari rating untuk user dan item tertentu
            user_rating['ratings'].append({
                'item': item,
                'rating': rating.rating if rating else 'N/A'  # Jika tidak ada rating, beri 'N/A'
            })
        user_ratings.append(user_rating)

    # Mengirimkan data ke template
    context = {
        'user_ratings': user_ratings,
        'items': items,
        'pelanggan_list': pelanggan_list,  # Tambahkan pelanggan_list ke context
    }
    return render(request, 'keloladata/keloladata.html', context)

def tambah_pelanggan(request):
    if request.method == 'POST':
        # Mengambil data dari form
        nama_pelanggan_baru = request.POST.get('nama_pelanggan_baru')
        no_telp_pelanggan = request.POST.get('no_telp_pelanggan')

        # Simpan pelanggan baru ke dalam database
        pelanggan_baru = User(name=nama_pelanggan_baru, no_telp=no_telp_pelanggan)
        pelanggan_baru.save()

        # Redirect kembali ke halaman keloladata
        return redirect('keloladata')
    
def kelolaitem(request):
     # Mengambil semua item untuk dijadikan header kolom
    items = Item.objects.all()

    context= {
        'items': items
    }
    
    return render(request, 'kelolaitem/kelolaitem.html', context)

def tambah_item(request):
    if request.method == 'POST':
        nama_item = request.POST.get('nama_item_baru')

        # Simpan pelanggan baru ke dalam database
        item_baru = Item(name=nama_item)
        item_baru.save()

        return redirect('kelolaitem')
    
def hapus_item(request, item_id):
    # Mengambil item berdasarkan ID atau tampilkan 404 jika tidak ditemukan
    item = get_object_or_404(Item, id=item_id)

    # Menghapus item
    item.delete()

    # Redirect kembali ke halaman kelolaitem setelah penghapusan
    return redirect('kelolaitem')

def edit_item(request, item_id):
    # Mengambil item berdasarkan ID atau tampilkan 404 jika tidak ditemukan
    item = get_object_or_404(Item, id=item_id)

    if request.method == 'POST':
        # Ambil data dari form
        nama_item = request.POST.get('nama_item')

        # Update nama item
        item.name = nama_item
        item.save()

        # Redirect kembali ke halaman kelolaitem setelah perubahan
        return redirect('kelolaitem')

def tambah_penilaian(request):
    if request.method == 'POST':
        # Ambil data pelanggan
        pelanggan_id = request.POST.get('pelanggan_id')
        pelanggan = get_object_or_404(User, id=pelanggan_id)

        # Ambil semua item
        items = Item.objects.all()

        # Simpan nilai rating untuk setiap item
        for item in items:
            rating_value = request.POST.get(f'item_{item.id}')
            if rating_value:
                rating_value = float(rating_value)
                
                # Cek apakah rating sudah ada, jika belum tambahkan
                rating, created = Rating.objects.get_or_create(user=pelanggan, item=item)
                
                # Update nilai rating
                rating.rating = rating_value
                rating.save()

        # Redirect kembali ke halaman keloladata setelah data ditambahkan
        return redirect('keloladata')

def get_ratings_for_pelanggan(request):
    pelanggan_id = request.GET.get('pelanggan_id')

    # Ambil pelanggan berdasarkan ID
    user = User.objects.get(id=pelanggan_id)

    # Ambil semua item
    items = Item.objects.all()

    # Buat dictionary untuk menyimpan nilai
    ratings_data = {}

    # Loop melalui semua item dan ambil rating untuk user tersebut
    for item in items:
        rating = Rating.objects.filter(user=user, item=item).first()  # Ambil rating
        ratings_data[item.id] = rating.rating if rating else ''  # Masukkan nilai jika ada

    return JsonResponse({'ratings': ratings_data})




