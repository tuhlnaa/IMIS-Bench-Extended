# [Linux]
# chmod +x ./script/Install_dependencies.sh
while read line; do
  # Skip empty lines and comments
  if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
    echo "Installing $line..."
    pip install $line
  fi
done < requirements.txt