import os
import json
import subprocess

try:
    import server
    import folder_paths
    web = server.web

    @server.PromptServer.instance.routes.get("/vsrfi/video_info")
    async def get_video_info(request):
        # Support both full filesystem path and ComfyUI filename
        full_path = request.query.get("path", "")
        filename = request.query.get("filename", "")

        if full_path:
            file_path = full_path
        elif filename:
            file_path = folder_paths.get_annotated_filepath(filename)
        else:
            return web.json_response({"error": "no filename or path"}, status=400)

        # Enforce that the file has a safe extension to prevent probing arbitrary system files
        allowed_extensions = {'.mp4', '.webm', '.mkv', '.mov', '.avi', '.gif'}
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in allowed_extensions:
            return web.json_response({"error": "forbidden extension"}, status=403)

        if not os.path.exists(file_path):
            return web.json_response({"error": "file not found"}, status=404)

        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,nb_frames',
                '-show_entries', 'format=duration',
                '-of', 'json',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            data = json.loads(result.stdout)

            stream = data.get('streams', [{}])[0]

            # Parse FPS from r_frame_rate (e.g. "30/1", "24000/1001")
            fps_str = stream.get('r_frame_rate', '30/1')
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 30.0

            # Frame count and duration
            nb_frames = int(stream.get('nb_frames', 0))
            duration = float(data.get('format', {}).get('duration', 0))

            # If nb_frames is 0, estimate from duration
            if nb_frames == 0 and duration > 0 and fps > 0:
                nb_frames = int(duration * fps)

            return web.json_response({
                'fps': round(fps, 3),
                'frame_count': nb_frames,
                'duration': round(duration, 3)
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @server.PromptServer.instance.routes.get("/vsrfi/view")
    async def view_video(request):
        """Serve a video file from an absolute filesystem path for preview."""
        file_path = request.query.get("path", "")
        if not file_path or not os.path.exists(file_path):
            return web.Response(status=404)

        content_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.gif': 'image/gif',
        }
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in content_types:
            return web.Response(status=403, text="Forbidden: Invalid file extension")

        content_type = content_types.get(ext, 'application/octet-stream')

        return web.FileResponse(file_path, headers={'Content-Type': content_type})

except ImportError:
    pass
