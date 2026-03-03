from supabase_client import upload_file

url = upload_file("image_bucket", "test.jpg")
print(url)