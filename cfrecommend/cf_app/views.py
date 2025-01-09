from .models import Rating, User, Item, Pesanan
from django.contrib.auth.models import User as AuthUser, Group  # Alias untuk Django User
from django.shortcuts import render
from django.shortcuts import get_object_or_404, redirect
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from django.http import JsonResponse
from django.contrib.auth import authenticate, login as auth_login
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q

@login_required(login_url='login')
def dashboard(request):
    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    # Ambil semua item
    items = Item.objects.all()

    # Ambil data rekomendasi (menggunakan logika sebelumnya)
    pesanan = Pesanan.objects.select_related('item', 'user').all()
    item_counts = pesanan.values('item__name').annotate(total=Count('item')).order_by('-total')

    rekomendasi = {}
    for item in item_counts:
        item_name = item['item__name']
        total_count = item['total']
        related_items = (
            pesanan.filter(user__in=pesanan.filter(item__name=item_name).values('user'))
            .exclude(item__name=item_name)
            .values('item__name')
            .annotate(total=Count('item'))
            .order_by('-total')
        )
        rekomendasi[item_name] = {
            'count': total_count,
            'related': [
                {'name': rel_item['item__name'], 'count': rel_item['total']}
                for rel_item in related_items
            ],
        }

    # Format data rekomendasi
    rekomendasi_list = [
        {
            'utama': key,
            'utama_count': value['count'],
            'pendamping': ', '.join(
                [f"{rel['name']} ({rel['count']})" for rel in value['related']]
            ),
        }
        for key, value in rekomendasi.items()
    ]

    context = {
        'is_admin': is_admin,
        'is_user': is_user,
        'items': items,  # Data untuk daftar semua item
        'rekomendasi_list': rekomendasi_list,  # Data untuk rekomendasi
    }
    return render(request, 'admin.html', context)

def login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Jika user sudah login, arahkan ke dashboard

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            return redirect('dashboard')  # Redirect ke halaman dashboard setelah login sukses
        else:
            messages.error(request, 'Username atau password salah.')  # Menampilkan pesan error

    return render(request, 'login.html')

def logout(request):
    auth_logout(request)
    return redirect('login')  # Arahkan kembali ke halaman login setelah logout

@login_required(login_url='login')
def home(request):
    return render(request, 'dashboard/home.html')

@login_required(login_url='login')
def kelolapesanan(request):
    # Mengambil semua user untuk dropdown pelanggan
    pelanggan_list = User.objects.all()

    # Mengambil semua pesanan untuk ditampilkan dalam tabel pesanan
    pesanan_list = Pesanan.objects.select_related('user', 'item').all()

    # Mengambil semua item untuk dropdown
    items = Item.objects.all()

    # Mengecek role pengguna
    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    # Mengirimkan data ke template
    context = {
        'items': items,
        'pelanggan_list': pelanggan_list,
        'pesanan_list': pesanan_list,
        'is_admin': is_admin,
        'is_user': is_user,
    }

    return render(request, 'kelolapesanan/kelolapesanan.html', context)

@login_required(login_url='login')
def kelolauser(request):
    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    # Ambil semua data user dari AuthUser
    users = AuthUser.objects.prefetch_related('groups').all()

    context = {
        'is_admin': is_admin,
        'is_user': is_user,
        'users': users,  # Gunakan AuthUser untuk konteks
    }
    return render(request, 'kelolauser/kelolauser.html', context)

@login_required(login_url='login')
def add_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        role = request.POST.get('role')

        # Validasi apakah username sudah ada
        if AuthUser.objects.filter(username=username).exists():
            messages.error(request, f'Username "{username}" sudah digunakan.')
            return redirect('kelolauser')

        # Buat user baru
        user = AuthUser.objects.create_user(username=username, email=email, password=password)

        # Tambahkan user ke grup sesuai role
        group = Group.objects.get(name=role)
        user.groups.add(group)

        messages.success(request, f'User "{username}" berhasil ditambahkan.')
        return redirect('kelolauser')

    return redirect('kelolauser')  # Redirect jika bukan POST

@login_required(login_url='login')
def edit_password(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(AuthUser, id=user_id)
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        if new_password != confirm_password:
            messages.error(request, 'Password baru dan konfirmasi tidak cocok.')
        else:
            user.set_password(new_password)
            user.save()
            messages.success(request, f'Password untuk user {user.username} berhasil diubah.')

    return redirect('kelolauser')  # Redirect ke halaman kelola user

@login_required(login_url='login')
def delete_user(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(AuthUser, id=user_id)
        user.delete()
        messages.success(request, f'User {user.username} berhasil dihapus.')

    return redirect('kelolauser')  # Redirect ke halaman kelola user


@login_required(login_url='login')
def hubungiadmin(request):
    return render(request, 'hubungiadmin/hubungiadmin.html')

@login_required(login_url='login')
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

    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    # Mengirimkan data ke template
    context = {
        'user_ratings': user_ratings,
        'items': items,
        'pelanggan_list': pelanggan_list,
        'is_admin': is_admin,
        'is_user': is_user,
    }
    return render(request, 'keloladata/keloladata.html', context)

@login_required(login_url='login')
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

@login_required(login_url='login')
def kelolaitem(request):
     # Mengambil semua item untuk dijadikan header kolom
    items = Item.objects.all()

    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    context= {
        'items': items,
        'is_admin': is_admin,
        'is_user': is_user,
    }
    
    return render(request, 'kelolaitem/kelolaitem.html', context)

@login_required(login_url='login')
def tambah_item(request):
    if request.method == 'POST':
        nama_item = request.POST.get('nama_item_baru')

        # Simpan pelanggan baru ke dalam database
        item_baru = Item(name=nama_item)
        item_baru.save()

        return redirect('kelolaitem')

@login_required(login_url='login')
def hapus_item(request, item_id):
    # Mengambil item berdasarkan ID atau tampilkan 404 jika tidak ditemukan
    item = get_object_or_404(Item, id=item_id)

    # Menghapus item
    item.delete()

    # Redirect kembali ke halaman kelolaitem setelah penghapusan
    return redirect('kelolaitem')

@login_required(login_url='login')
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

@login_required(login_url='login')
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

@login_required(login_url='login')
def tambah_pesanan(request):
    if request.method == 'POST':
        # Ambil data pelanggan
        pelanggan_id = request.POST.get('pelanggan_id')
        pelanggan = get_object_or_404(User, id=pelanggan_id)

        # Ambil item_id dari form
        item_id = request.POST.get('item_id')
        item = get_object_or_404(Item, id=item_id)

        # Simpan pesanan ke database
        pesanan = Pesanan(user=pelanggan, item=item)
        pesanan.save()

        # Redirect ke halaman daftar pesanan
        return redirect('kelolapesanan')

    # Jika request bukan POST, kembalikan ke halaman sebelumnya
    return redirect('kelolapesanan')


@login_required(login_url='login')
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

@login_required(login_url='login')
def hasilrekomendasi(request):
    # Ambil data dari database
    ratings = Rating.objects.all()

    # Konversi data dari database menjadi DataFrame
    data_dict = {
        'user_id': [rating.user.id for rating in ratings],
        'item_id': [rating.item.id for rating in ratings],
        'rating': [rating.rating for rating in ratings]
    }
    
    df = pd.DataFrame(data_dict)

    # Jika tidak ada data, kembalikan respons kosong
    if df.empty:
        context = {
            "all_recommendations": [],
            "mae": None
        }
        return render(request, 'hasilrekomendasi/hasilrekomendasi.html', context)

    # Pivot tabel untuk membuat matriks user-item
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')

    # Mengganti NaN dengan 0, karena cosine similarity tidak bisa menangani nilai NaN
    user_item_matrix.fillna(0, inplace=True)

    # Menghitung kemiripan antar item menggunakan cosine similarity
    item_similarity = cosine_similarity(user_item_matrix.T)  # Transpose untuk menghitung antar item
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    # List untuk menyimpan hasil rekomendasi dan nilai prediksi serta rating aktual
    all_recommendations = []
    all_actual_ratings = []
    all_predicted_ratings = []

    for user_id in df['user_id'].unique():
        # Item yang sudah di-rating oleh user
        rated_items = df[df['user_id'] == user_id]
        user_rated_items = rated_items['item_id'].tolist()

        # Mencari item yang mirip dengan item yang sudah di-rating
        not_rated = [item_id for item_id in user_item_matrix.columns if item_id not in user_rated_items]
        
        # Untuk setiap item yang belum di-rating, prediksi skor berdasarkan item-item serupa
        recommendations = []
        for item in not_rated:
            # Mengambil item-item yang sudah di-rating oleh user
            similar_items = item_similarity_df[item].sort_values(ascending=False)

            # Menghitung skor prediksi: rata-rata dari rating item-item yang mirip
            weighted_sum = 0
            sim_sum = 0
            for similar_item, similarity in similar_items.items():
                if similar_item in user_rated_items:
                    rating = rated_items[rated_items['item_id'] == similar_item]['rating'].values[0]
                    weighted_sum += similarity * rating
                    sim_sum += similarity

            # Jika tidak ada kesamaan, prediksi 0
            predicted_rating = weighted_sum / sim_sum if sim_sum != 0 else 0

            recommendations.append({
                'user_id': user_id,
                'item_id': item,
                'pred_score': predicted_rating
            })

        # Simpan hasil rekomendasi untuk user ini
        all_recommendations.extend(recommendations)

        # Simpan rating aktual dan prediksi untuk menghitung MAE
        for _, row in rated_items.iterrows():
            actual_rating = row['rating']
            similar_items = item_similarity_df[row['item_id']].sort_values(ascending=False)
            weighted_sum = 0
            sim_sum = 0
            for similar_item, similarity in similar_items.items():
                if similar_item in user_rated_items:
                    rating = rated_items[rated_items['item_id'] == similar_item]['rating'].values[0]
                    weighted_sum += similarity * rating
                    sim_sum += similarity
            predicted_rating = weighted_sum / sim_sum if sim_sum != 0 else 0

            all_actual_ratings.append(actual_rating)
            all_predicted_ratings.append(predicted_rating)

    # Menghitung MAE (Mean Absolute Error)
    mae = mean_absolute_error(all_actual_ratings, all_predicted_ratings)

    # Mengubah hasil rekomendasi menjadi DataFrame untuk kemudahan manipulasi
    all_recommendations_df = pd.DataFrame(all_recommendations)

    # Mengambil nama user dan nama item dari database berdasarkan rekomendasi
    all_recommendations_df['user_name'] = all_recommendations_df['user_id'].apply(
        lambda x: User.objects.get(id=x).name
    )
    all_recommendations_df['item_name'] = all_recommendations_df['item_id'].apply(
        lambda x: Item.objects.get(id=x).name
    )

    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()
    # Mengirim hasil ke template HTML
    context = {
        "all_recommendations": all_recommendations_df.to_dict(orient='records'),
        "mae": mae,
        'is_admin': is_admin,
        'is_user': is_user,
    }
    
    return render(request, 'hasilrekomendasi/hasilrekomendasi.html', context)

@login_required(login_url='login')
def rekomendasiuser(request):
    # Ambil semua pesanan dari database
    pesanan = Pesanan.objects.select_related('item', 'user').all()

    # Buat DataFrame dengan data pengguna dan item
    data = pd.DataFrame(list(pesanan.values('user_id', 'item_id')))
    
    # Buat matriks user-item
    user_item_matrix = data.pivot_table(index='user_id', columns='item_id', aggfunc='size', fill_value=0)
    
    # Hitung kesamaan antar item menggunakan cosine similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    # Rekomendasi item berdasarkan item yang sering dipesan oleh user
    rekomendasi = {}
    for item_id in user_item_matrix.columns:
        # Cari nama item utama dan jumlah pesanannya
        utama_name = Pesanan.objects.filter(item_id=item_id).first().item.name
        utama_count = pesanan.filter(item_id=item_id).count()
        
        # Cari item yang paling mirip berdasarkan nilai kesamaan
        similar_items = item_similarity_df[item_id].sort_values(ascending=False)[1:6]  # Ambil 5 item paling mirip
        rekomendasi[item_id] = {
            'utama': utama_name,
            'utama_count': utama_count,  # Tambahkan jumlah pesanan item utama
            'pendamping': [
                {
                    'name': Pesanan.objects.filter(item_id=sim_item).first().item.name,
                    'count': pesanan.filter(item_id=sim_item).count(),  # Jumlah pesanan item pendamping
                }
                for sim_item, similarity in similar_items.items() if similarity > 0.0  # Filter similarity > 0.0
            ]
        }

    # Konversi rekomendasi menjadi format untuk template
    rekomendasi_list = [
        {
            'utama': f"{value['utama']} ({value['utama_count']})",  # Tambahkan jumlah pesanan utama ke string
            'pendamping': ', '.join(
                [f"{rel['name']} ({rel['count']})" for rel in value['pendamping']]  # Tampilkan jumlah pesanan pendamping
            ),
        }
        for key, value in rekomendasi.items()
    ]

    is_admin = request.user.groups.filter(name="Admin").exists()
    is_user = request.user.groups.filter(name="User").exists()

    context = {
        'rekomendasi_list': rekomendasi_list,
        'is_admin': is_admin,
        'is_user': is_user,
    }

    return render(request, 'rekomendasiuser/rekomendasiuser.html', context)



@login_required(login_url='login')
# Fungsi untuk mengedit penilaian
def edit_penilaian(request, pelanggan_id):
    pelanggan = get_object_or_404(User, id=pelanggan_id)
    items = Item.objects.all()

    if request.method == 'POST':
        for item in items:
            rating_value = request.POST.get(f'item_{item.id}')

            # Cek apakah nilai dihapus (kosong)
            if rating_value == '':
                # Jika kosong, hapus rating untuk item tersebut
                Rating.objects.filter(user=pelanggan, item=item).delete()
            else:
                rating_value = float(rating_value)
                rating, created = Rating.objects.get_or_create(user=pelanggan, item=item)
                rating.rating = rating_value
                rating.save()

        return redirect('keloladata')

    # Ambil penilaian untuk pelanggan
    user_ratings = {}
    for item in items:
        rating = Rating.objects.filter(user=pelanggan, item=item).first()
        user_ratings[item.id] = rating.rating if rating else None

    context = {
        'pelanggan': pelanggan,
        'items': items,
        'user_ratings': user_ratings,
    }
    return render(request, 'edit_penilaian.html', context)

@login_required(login_url='login')
# Fungsi untuk menghapus pelanggan beserta semua penilaiannya
def hapus_pelanggan(request, pelanggan_id):
    if request.method == 'POST':
        # Ambil pelanggan berdasarkan pelanggan_id
        pelanggan = get_object_or_404(User, id=pelanggan_id)

        # Hapus semua rating terkait pelanggan ini
        Rating.objects.filter(user=pelanggan).delete()

        # Hapus pelanggan itu sendiri
        pelanggan.delete()

        # Redirect kembali ke halaman keloladata setelah pelanggan dihapus
        return redirect('keloladata')
