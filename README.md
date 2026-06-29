# Plataforma Cloud-Native de Búsqueda Semántica en PDFs

Proyecto base para el examen final de Computación de Alto Desempeño y Cloud Computing.

Arquitectura utilizada:

- **Streamlit** como interfaz web.
- **Docker** para contenerizar la aplicación.
- **MongoDB Atlas** como base NoSQL para almacenar chunks y embeddings.
- **Cohere** para generar embeddings.
- **Gemini** para generar respuestas con contexto.
- **GitHub Actions** para CI/CD.
- **Azure Web App for Container** para el despliegue.
- **Azure Container Registry** para publicar la imagen Docker.

---

## 1. Variables de entorno

Copia `.env.example` como `.env` y completa los valores reales:

```bash
cp .env.example .env
```

Variables necesarias:

```bash
APP_USER="Apellido Nombre"
MONGO_URI="mongodb+srv://..."
MONGODB_DB="hpc_pdf_analytics"
MONGODB_COLLECTION="pdf_chunks"
MONGO_VECTOR_INDEX="vector_index"
COHERE_API_KEY="..."
GEMINI_API_KEY="..."
COHERE_MODEL="embed-multilingual-v3.0"
GEMINI_MODEL="gemini-1.5-flash"
```

No subas el archivo `.env` a GitHub.

---

## 2. Ejecución local sin Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

---

## 3. Ejecución local con Docker

```bash
docker build -t hpc-pdf-chatbot .
docker run --env-file .env -p 8501:8501 hpc-pdf-chatbot
```

Abrir:

```text
http://localhost:8501
```

---

## 4. Índice vectorial en MongoDB Atlas

En MongoDB Atlas, entra a tu clúster, luego a Atlas Search / Vector Search, y crea un índice llamado:

```text
vector_index
```

Usa el contenido del archivo `mongo_vector_index.json`.

El modelo `embed-multilingual-v3.0` de Cohere genera vectores de 1024 dimensiones, por eso el índice usa:

```json
"numDimensions": 1024
```

Si el índice todavía no está listo, la app puede funcionar con un fallback local de similitud coseno para fines de validación.

---

## 5. Crear repositorio público en GitHub

```bash
git init
git add .
git commit -m "Initial cloud-native PDF chatbot app"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

---

## 6. Publicar imagen en Azure Container Registry

Ejemplo con Azure CLI:

```bash
az login
az group create --name rg-hpc-final --location eastus
az acr create --resource-group rg-hpc-final --name acrhpcfinalXXX --sku Basic --admin-enabled true
az acr login --name acrhpcfinalXXX

docker build -t hpc-pdf-chatbot .
docker tag hpc-pdf-chatbot acrhpcfinalXXX.azurecr.io/hpc-pdf-chatbot:latest
docker push acrhpcfinalXXX.azurecr.io/hpc-pdf-chatbot:latest
```

Reemplaza `XXX` por tus iniciales o un sufijo único.

---

## 7. Crear Azure Web App for Container

```bash
az appservice plan create \
  --name asp-hpc-final \
  --resource-group rg-hpc-final \
  --is-linux \
  --sku B1

az webapp create \
  --resource-group rg-hpc-final \
  --plan asp-hpc-final \
  --name webapp-hpc-final-XXX \
  --deployment-container-image-name acrhpcfinalXXX.azurecr.io/hpc-pdf-chatbot:latest
```

Configura el puerto de Streamlit y las variables:

```bash
az webapp config appsettings set \
  --resource-group rg-hpc-final \
  --name webapp-hpc-final-XXX \
  --settings \
  WEBSITES_PORT=8501 \
  APP_USER="Apellido Nombre" \
  MONGO_URI="mongodb+srv://..." \
  MONGODB_DB="hpc_pdf_analytics" \
  MONGODB_COLLECTION="pdf_chunks" \
  MONGO_VECTOR_INDEX="vector_index" \
  COHERE_API_KEY="..." \
  GEMINI_API_KEY="..." \
  COHERE_MODEL="embed-multilingual-v3.0" \
  GEMINI_MODEL="gemini-1.5-flash"
```

---

## 8. GitHub Actions

El workflow está en:

```text
.github/workflows/azure-webapp-container.yml
```

Configura estos secrets en GitHub:

```text
ACR_LOGIN_SERVER
ACR_USERNAME
ACR_PASSWORD
AZURE_WEBAPP_NAME
AZURE_WEBAPP_PUBLISH_PROFILE
```

Puedes obtener credenciales del ACR con:

```bash
az acr credential show --name acrhpcfinalXXX
```

El publish profile se descarga desde Azure Portal en la Web App: **Get publish profile**.

---

## 9. Cambio visual para validar CI/CD

Edita en `app.py` esta línea del sidebar:

```python
st.markdown("**Cambio visual CI/CD:** v1.0 - Examen Final HPC")
```

Por ejemplo:

```python
st.markdown("**Cambio visual CI/CD:** v2.0 - Despliegue automatizado validado")
```

Luego ejecuta:

```bash
git add app.py
git commit -m "Visual change for GitHub Actions validation"
git push
```

Evidencias sugeridas:

1. Captura del commit en GitHub.
2. Captura del workflow ejecutado correctamente.
3. Captura de la app desplegada mostrando el cambio visual.

---

## 10. Validación del chatbot

1. Abre la aplicación desplegada.
2. Sube un PDF.
3. Haz clic en **Procesar PDF**.
4. Realiza una pregunta relacionada al contenido.
5. Captura la pregunta, la respuesta y el contexto recuperado.
