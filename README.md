Installing Guide
1. Install Python (www.python.org)
2. buat database di MySQL dengan db "db_cf"
3. import database
4. clone file project ini
5. buka folder cf-recom melalui command prompt
6. aktifkan env dengan mengetik myenv\Scripts\activate
7. jika env aktif, akan tampil tulisan (env) pada nama folder di cmd
8. setelah env aktif, ketik "cd cfrecommend"
9. aktifkan server dengan mengetik "py manage.py runserver"
10. jika muncul error library belum terinstall seperti "No module named 'pandas' ", install library yang belum diinstall dengan cara ketik "pip install pandas"
11. lakukan hal serupa jika masih ada error no module found
12. jika server berhasil berjalan, akan ditandai dengan salah satu baris dengan keterangan "Starting development server at http://127.0.0.1:8000/"
13. buka browser lalu ketik "localhost:8000"
14. login dengan username "admin" dan password "admin"
