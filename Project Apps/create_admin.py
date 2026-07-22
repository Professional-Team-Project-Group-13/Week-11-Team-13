from backend import auth


USERNAME = "admin2"          
PASSWORD = "ChangeMe123!"    
FULL_NAME = "Francis Administrator"  
# --------------------------------------

def main():
    success, message = auth.create_user(
        username=USERNAME,
        password=PASSWORD,
        role="admin",
        full_name=FULL_NAME,
        status="active"
    )
    if success:
        print(f"Administrator account created successfully.：{USERNAME}")
        print(f"   password：{PASSWORD}（Please log in and make changes as soon as possible.）")
    else:
        print(f"Creation failed：{message}")

if __name__ == "__main__":
    main()