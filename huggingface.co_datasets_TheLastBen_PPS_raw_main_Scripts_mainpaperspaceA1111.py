import os
from IPython.display import clear_output
from subprocess import call, getoutput, run
import time
import sys
import fileinput
import ipywidgets as widgets
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import re

from urllib.request import urlopen, Request
import tempfile
from tqdm import tqdm 



def Deps(force_reinstall):

    if not force_reinstall and os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
        ntbk()
        print('[1;32mModules and notebooks updated, dependencies already installed')

    else:
        call("pip install --root-user-action=ignore --no-deps -q accelerate==0.12.0", shell=True, stdout=open('/dev/null', 'w'))
        if not os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
            os.chdir('/usr/local/lib/python3.10/dist-packages')
            call("rm -r torch torch-1.12.1+cu116.dist-info torchaudio* torchvision* PIL Pillow* transformers* numpy* gdown*", shell=True, stdout=open('/dev/null', 'w'))
        ntbk()
        if not os.path.exists('/models'):
            call('mkdir /models', shell=True)
        if not os.path.exists('/kaggle/working/models'):
            call('ln -s /models /kaggle/working/', shell=True)
        if os.path.exists('/deps'):
            call("rm -r /deps", shell=True)
        call('mkdir /deps', shell=True)
        if not os.path.exists('cache'):
            call('mkdir cache', shell=True)
        os.chdir('/deps')
        call('wget -q -i https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dependencies/aptdeps.txt', shell=True)
        call('dpkg -i *.deb', shell=True, stdout=open('/dev/null', 'w'))
        depsinst("https://huggingface.co/TheLastBen/dependencies/resolve/main/ppsdeps.tar.zst", "/deps/ppsdeps.tar.zst")
        call('tar -C / --zstd -xf ppsdeps.tar.zst', shell=True, stdout=open('/dev/null', 'w'))
        call("sed -i 's@~/.cache@/kaggle/working//cache@' /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", shell=True)
        os.chdir('/kaggle/working/')
        call("git clone --depth 1 -q --branch main https://github.com/TheLastBen/diffusers /diffusers", shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
        call('pip install --root-user-action=ignore --disable-pip-version-check -qq tomesd', shell=True, stdout=open('/dev/null', 'w'))
        if not os.path.exists('/kaggle/working/diffusers'):
            call('ln -s /diffusers /kaggle/working/', shell=True)
        call("rm -r /deps", shell=True)
        os.chdir('/kaggle/working')
        clear_output()

        done()



def depsinst(url, dst):
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    with tqdm(total=file_size, disable=False, mininterval=0.5,
              bar_format='Installing dependencies |{bar:20}| {percentage:3.0f}%') as pbar:
        with open(dst, "wb") as f:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
            f.close()


def ntbk():

    os.chdir('/kaggle/working')
    if not os.path.exists('Latest_Notebooks'):
        call('mkdir Latest_Notebooks', shell=True)
    else:
        call('rm -r Latest_Notebooks', shell=True)
        call('mkdir Latest_Notebooks', shell=True)
    os.chdir('/kaggle/working//Latest_Notebooks')
    call('wget -q -i https://huggingface.co/datasets/TheLastBen/PPS/raw/main/Notebooks.txt', shell=True)
    call('rm Notebooks.txt', shell=True)
    os.chdir('/kaggle/working')


def repo():

    print('[1;32mInstalling/Updating the repo...')
    os.chdir('/kaggle/working')
    if not os.path.exists('/kaggle/working/sd/stablediffusion'):
       call('wget -q -O sd_rep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_rep.tar.zst', shell=True)
       call('tar --zstd -xf sd_rep.tar.zst', shell=True)
       call('rm sd_rep.tar.zst', shell=True)        

    os.chdir('/kaggle/working/sd')
    if not os.path.exists('stable-diffusion-webui'):
        call('git clone -q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui', shell=True)

    os.chdir('/kaggle/working/sd/stable-diffusion-webui/')
    call('git reset --hard', shell=True, stdout=open('/dev/null', 'w'))
    print('[1;32m')
    call('git pull', shell=True, stdout=open('/dev/null', 'w'))
    os.chdir('/kaggle/working')
    clear_output()
    done()


def mdl(Original_Model_Version, Path_to_MODEL, MODEL_LINK, safetensors, Temporary_Storage):
    import gdown
    if Path_to_MODEL !='':
      if os.path.exists(str(Path_to_MODEL)):
        print('[1;32mUsing the trained model.')
        model=Path_to_MODEL
      else:
          print('[1;31mWrong path, check that the path to the model is correct')

    elif MODEL_LINK != "":
      modelname="model.safetensors" if safetensors else "model.ckpt"
      if Temporary_Storage:
         model=f'/models/{modelname}'
      else:
         model=f'/kaggle/working/sd/stable-diffusion-webui/models/Stable-diffusion/{modelname}'
      if os.path.exists(model):
        call('rm '+model, shell=True)
      gdown.download(url=MODEL_LINK, output=model, quiet=False, fuzzy=True)

      if os.path.exists(model) and os.path.getsize(model) > 1810671599:
        clear_output()
        print('[1;32mModel downloaded, using the trained model.')
      else:
        print('[1;31mWrong link, check that the link is valid')

    else:
        if Original_Model_Version == "v1.5":
           model="/datasets/stable-diffusion-classic/SDv1.5.ckpt"
           print('[1;32mUsing the original V1.5 model')
        elif Original_Model_Version == "v2-512":
           model="/datasets/stable-diffusion-v2-1-base-diffusers/stable-diffusion-2-1-base/v2-1_512-nonema-pruned.safetensors"
           print('[1;32mUsing the original V2-512 model')
        elif Original_Model_Version == "v2-768":
           model="/datasets/stable-diffusion-v2-1/stable-diffusion-2-1/v2-1_768-nonema-pruned.safetensors"
           print('[1;32mUsing the original V2-768 model')
        else:
            model=""
            print('[1;31mWrong model version')
    try:
        model
    except:
        model="/kaggle/working/sd/stable-diffusion-webui/models/Stable-diffusion"    

    return model


def CN(ControlNet_Model, ControlNet_v2_Model):
    
    def download(url, model_dir):

        filename = os.path.basename(urlparse(url).path)
        pth = os.path.abspath(os.path.join(model_dir, filename))
        if not os.path.exists(pth):
            print('Downloading: '+os.path.basename(url))
            download_url_to_file(url, pth, hash_prefix=None, progress=True)
        else:
          print(f"[1;32mThe model {filename} already exists[0m")    

    wrngv1=False
    os.chdir('/kaggle/working/sd/stable-diffusion-webui/extensions')
    if not os.path.exists("sd-webui-controlnet"):
      call('git clone https://github.com/Mikubill/sd-webui-controlnet.git', shell=True)
      os.chdir('/kaggle/working')
    else:
      os.chdir('sd-webui-controlnet')
      call('git reset --hard', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('git pull', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      os.chdir('/kaggle/working')

    mdldir="/kaggle/working/sd/stable-diffusion-webui/extensions/sd-webui-controlnet/models"
    for filename in os.listdir(mdldir):
      if "_sd14v1" in filename:
        renamed = re.sub("_sd14v1", "-fp16", filename)
        os.rename(os.path.join(mdldir, filename), os.path.join(mdldir, renamed))

    call('wget -q -O CN_models.txt https://github.com/TheLastBen/fast-stable-diffusion/raw/main/AUTOMATIC1111_files/CN_models.txt', shell=True)
    call('wget -q -O CN_models_v2.txt https://github.com/TheLastBen/fast-stable-diffusion/raw/main/AUTOMATIC1111_files/CN_models_v2.txt', shell=True)
    
    with open("CN_models.txt", 'r') as f:
        mdllnk = f.read().splitlines()
    with open("CN_models_v2.txt", 'r') as d:
        mdllnk_v2 = d.read().splitlines()
    call('rm CN_models.txt CN_models_v2.txt', shell=True)
    
    cfgnames=[os.path.basename(url).split('.')[0]+'.yaml' for url in mdllnk_v2]
    os.chdir('/kaggle/working/sd/stable-diffusion-webui/extensions/sd-webui-controlnet/models')
    for name in cfgnames:
        run(['cp', 'cldm_v21.yaml', name])
    os.chdir('/kaggle/working')    

    if ControlNet_Model == "All" or ControlNet_Model == "all" :     
      for lnk in mdllnk:
          download(lnk, mdldir)
      clear_output()

      
    elif ControlNet_Model == "15":
      mdllnk=list(filter(lambda x: 't2i' in x, mdllnk))
      for lnk in mdllnk:
          download(lnk, mdldir)
      clear_output()        


    elif ControlNet_Model.isdigit() and int(ControlNet_Model)-1<14 and int(ControlNet_Model)>0:
      download(mdllnk[int(ControlNet_Model)-1], mdldir)
      clear_output()
      
    elif ControlNet_Model == "none":
       pass
       clear_output()

    else:
      print('[1;31mWrong ControlNet V1 choice, try again')
      wrngv1=True

    if ControlNet_v2_Model == "All" or ControlNet_v2_Model == "all" :
      for lnk_v2 in mdllnk_v2:
          download(lnk_v2, mdldir)
      if not wrngv1:
        clear_output()
      done()

    elif ControlNet_v2_Model.isdigit() and int(ControlNet_v2_Model)-1<5:
      download(mdllnk_v2[int(ControlNet_v2_Model)-1], mdldir)
      if not wrngv1:
        clear_output()
      done()
    
    elif ControlNet_v2_Model == "none":
       pass
       if not wrngv1:
         clear_output()
       done()       

    else:
      print('[1;31mWrong ControlNet V2 choice, try again')




def sdui(User, Password, model):

    auth=f"--gradio-auth {User}:{Password}"
    if User =="" or Password=="":
      auth=""

    
    call('wget -q -O /usr/local/lib/python3.10/dist-packages/gradio/blocks.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/blocks.py', shell=True)
    
    localurl="tensorboard-"+os.environ.get('PAPERSPACE_FQDN')
    
    for line in fileinput.input('/usr/local/lib/python3.10/dist-packages/gradio/blocks.py', inplace=True):
      if line.strip().startswith('self.server_name ='):
          line = f'            self.server_name = "{localurl}"\n'
      if line.strip().startswith('self.protocol = "https"'):
          line = '            self.protocol = "https"\n'
      if line.strip().startswith('if self.local_url.startswith("https") or self.is_colab'):
          line = ''
      if line.strip().startswith('else "http"'):
          line = ''
      sys.stdout.write(line)

     
    os.chdir('/kaggle/working/sd/stable-diffusion-webui/modules')
    call('wget -q -O paths.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/paths.py', shell=True)    
    call("sed -i 's@/content/gdrive/MyDrive/sd/stablediffusion@/kaggle/working/sd/stablediffusion@' /kaggle/working/sd/stable-diffusion-webui/modules/paths.py", shell=True)
    call("sed -i 's@\"quicksettings\": OptionInfo(.*@\"quicksettings\": OptionInfo(\"sd_model_checkpoint,  sd_vae, CLIP_stop_at_last_layers, inpainting_mask_weight, initial_noise_multiplier\", \"Quicksettings list\"),@' /kaggle/working/sd/stable-diffusion-webui/modules/shared.py", shell=True)
    os.chdir('/kaggle/working/sd/stable-diffusion-webui')
    clear_output()


    if model=="":
        mdlpth=""
    else:
        if os.path.isfile(model):
            mdlpth="--ckpt "+model
        else:
            mdlpth="--ckpt-dir "+model


    configf="--disable-console-progressbars --no-gradio-queue --no-half-vae --disable-safe-unpickle --api --no-download-sd-model --xformers --enable-insecure-extension-access --port 6006 --listen --skip-version-check "+auth+" "+mdlpth

    return configf    
    
    
def done():
    done = widgets.Button(
        description='Done!',
        disabled=True,
        button_style='success',
        tooltip='',
        icon='check'
    )
    display(done)
