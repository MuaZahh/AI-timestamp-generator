# 🚀 Deployment Guide: GitHub + Vercel

## Quick Deploy to Vercel

### 1. Push to GitHub

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: AI Timestamp Generator"

# Push to GitHub
git remote add origin https://github.com/yourusername/ai-timestamp-generator.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Vercel

1. **Connect to Vercel**:
   - Go to https://vercel.com
   - Sign up/login with GitHub
   - Click "Import Project"
   - Select your repository

2. **Environment Variables** (CRITICAL - Set these in Vercel dashboard):
   ```
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   SECRET_KEY=your-production-secret-key
   LOG_LEVEL=INFO
   DATABASE_URL=sqlite:///timestamp_generator.db
   MAX_CONTENT_LENGTH=104857600
   UPLOAD_FOLDER=uploads
   ```

3. **Deploy Settings**:
   - Framework: `Other`
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: Leave empty
   - Install Command: Leave empty

### 3. Post-Deployment

Your app will be live at: `https://your-app-name.vercel.app`

## ✅ Vercel Features Included:

- **FFmpeg**: ✅ Built-in (no manual installation needed)
- **File Uploads**: ✅ Supported
- **Database**: ✅ SQLite works on Vercel
- **Background Processing**: ⚠️ Limited (300s timeout)
- **Caching**: ✅ File system cache works
- **Batch Processing**: ⚠️ Limited by serverless constraints

## 🔧 Serverless Optimizations Made:

1. **Timeout Handling**: 300s max function duration
2. **Memory Optimization**: Efficient processing
3. **Cache Strategy**: Persistent file cache
4. **Error Handling**: Comprehensive error management
5. **Resource Management**: Automatic cleanup

## 🌟 Alternative: Full Server Deployment

For unlimited processing time and full batch capabilities, consider:

- **Railway**: https://railway.app (Easy Python deployment)
- **Render**: https://render.com (Free tier available) 
- **DigitalOcean App Platform**: Full featured
- **AWS/GCP/Azure**: Enterprise scale

## 📊 Performance Expectations:

### Vercel (Serverless):
- ✅ Fast cold starts
- ✅ Auto-scaling
- ✅ 5-minute video processing
- ⚠️ 300s timeout limit

### Full Server:
- ✅ Unlimited processing time
- ✅ Full batch processing
- ✅ Large video files
- ✅ Background jobs

## 🔒 Security Notes:

- Environment variables are secure in Vercel
- API keys are encrypted
- HTTPS enabled by default
- No sensitive data in repository

## 🐛 Troubleshooting:

### Common Issues:
1. **Build fails**: Check Python version (3.11.9)
2. **Import errors**: Verify requirements.txt
3. **API errors**: Check Deepgram API key
4. **Timeout**: Videos >5 minutes may timeout

### Solutions:
- Use smaller video files for testing
- Check Vercel function logs
- Verify environment variables
- Test locally first

Your AI Timestamp Generator is ready for production! 🎉