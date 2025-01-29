from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('home/', views.home, name='home'),
    path('kelolauser/', views.kelolauser, name='kelolauser'),
    path('kelolauser/add-user/', views.add_user, name='add_user'),
    path('kelolauser/edit-password/<int:user_id>/', views.edit_password, name='edit_password'),
    path('kelolauser/delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('hubungiadmin/', views.kelolauser, name='hubungiadmin'),
    path('keloladata/', views.keloladata, name='keloladata'),
    path('tambah-pelanggan/', views.tambah_pelanggan, name='tambah_pelanggan'),
    path('kelolaitem/', views.kelolaitem, name='kelolaitem'),
    path('tambah-item/', views.tambah_item, name='tambah_item'),
    path('tambah-penilaian/', views.tambah_penilaian, name='tambah_penilaian'),
    path('hapus-item/<int:item_id>/', views.hapus_item, name='hapus_item'),
    path('edit-item/<int:item_id>/', views.edit_item, name='edit_item'),
    path('get-ratings-for-pelanggan/', views.get_ratings_for_pelanggan, name='get_ratings_for_pelanggan'),
    path('hasilrekomendasi/', views.hasilrekomendasi, name='hasilrekomendasi'),
    path('edit-penilaian/<int:pelanggan_id>/', views.edit_penilaian, name='edit_penilaian'),
    path('hapus-pelanggan/<int:pelanggan_id>/', views.hapus_pelanggan, name='hapus_pelanggan'),
    path('kelolapesanan/', views.kelolapesanan, name='kelolapesanan'),
    path('tambah-pesanan/', views.tambah_pesanan, name='tambah_pesanan'),
    path('rekomendasiuser/', views.rekomendasiuser, name='rekomendasiuser'),
    path('pilihitem/', views.pilihitem, name='pilihitem'),
    path('rekomendasipelayan/', views.rekomendasipelayan, name='rekomendasipelayan'),
    path('stateofart/', views.stateofart, name='stateofart')
]