from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1hkSFPRFnw5hmkAnUnrDCOU8s0Wdc6I-T'
export_file_name = 'export.pkl'

classes = ['Ahead only', 'Beware of ice/snow', 'Bicycles crossing', 'Bumpy road', 'Children crossing', 'Dangerous curve to the left', 
	'Dangerous curve to the right', 'Double curve', 'End of all speed and passing limits', 'End of no passing', 
	'End of no passing by vehicles over 3.5 metric tons', 'End of speed limit (80km/h)', 'General caution', 'Go straight or left', 
	'Go straight or right', 'Keep left', 'Keep right', 'No entry', 'No passing', 'No passing for vehicles over 3.5 metric tons',
 	'No vehicles', 'Pedestrians', 'Priority road', 'Right-of-way at the next intersection', 'Road narrows on the right',
	'Road work', 'Roundabout mandatory', 'Slippery road', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'Speed limit (20km/h)',
	'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
 	'Stop', 'Traffic signals', 'Turn left ahead', 'Turn right ahead', 'Vehicles over 3.5 metric tons prohibited',
	'Wild animals crossing', 'Yield']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction_info = learn.predict(img)
    prediction = prediction_info[0]
    top_probs, top_classes = prediction_info[2].topk(5)
    # if probability is below 0.5 then it's not an image
    if top_probs[0].item() < 0.85:
        return JSONResponse({'result': 'wrong'})
    else:
        return JSONResponse({'result': str(prediction)})

@app.route('/about')
def about_page(request):
    html = path/'view'/'about.html'
    return HTMLResponse(html.open().read())

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
