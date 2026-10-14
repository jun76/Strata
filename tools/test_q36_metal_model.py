"""Whole-model Metal vs NumPy on a synthetic four-layer Qwen35-MoE GGUF.

No model download. Requires the pinned gguf-py checkout and build-metal/strata-q36-metal.
Covers full forward wiring, selected experts, recurrent steps and the Strata pipe protocol.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('STRATA_GGUF_PY', str(ROOT/'third_party/llama.cpp/gguf-py')))
from gguf import GGUFWriter
N,FF,V,NE,K,D,C=64,32,64,4,2,128,384
rng=np.random.default_rng(41)
weights={}
def add(name,shape):
    a=rng.normal(0,.04,shape).astype(np.float32)
    if 'norm.weight' in name:a[:]=1
    if name.endswith('ssm_a'):a[:]=-.5
    weights[name]=a
add('token_embd.weight',(V,N));add('output.weight',(V,N));add('output_norm.weight',(N,))
for l in range(4):
    p=f'blk.{l}.'
    for name,shape in {'attn_norm.weight':(N,), 'post_attention_norm.weight':(N,),
        'ffn_gate_inp.weight':(NE,N),'ffn_gate_inp_shexp.weight':(N,),
        'ffn_gate_exps.weight':(NE,FF,N),'ffn_up_exps.weight':(NE,FF,N),'ffn_down_exps.weight':(NE,N,FF),
        'ffn_gate_shexp.weight':(FF,N),'ffn_up_shexp.weight':(FF,N),'ffn_down_shexp.weight':(N,FF)}.items():add(p+name,shape)
    specific=({'attn_qkv.weight':(C,N),'attn_gate.weight':(D,N),'ssm_out.weight':(N,D),
               'ssm_conv1d.weight':(C,4),'ssm_alpha.weight':(1,N),'ssm_beta.weight':(1,N),
               'ssm_a':(1,),'ssm_dt.bias':(1,),'ssm_norm.weight':(D,)} if l<3 else
              {'attn_q.weight':(512,N),'attn_k.weight':(256,N),'attn_v.weight':(256,N),
               'attn_output.weight':(N,256),'attn_q_norm.weight':(256,),'attn_k_norm.weight':(256,)})
    for name,shape in specific.items():add(p+name,shape)

def rms(x,w):return x/np.sqrt(np.mean(x*x)+1e-6)*w
def silu(x):return x/(1+np.exp(-x))
def sigmoid(x):return 1/(1+np.exp(-x))
def cpu(ids):
    hist=[np.zeros((3,C),np.float32) for _ in range(3)]
    states=[np.zeros((D,D),np.float32) for _ in range(3)]
    kc=[];vc=[];logits=[]
    for pos,tok in enumerate(ids):
        x=weights['token_embd.weight'][tok].copy()
        for l in range(4):
            def w(name):return weights[f'blk.{l}.'+name]
            h=rms(x,w('attn_norm.weight'))
            if l<3:
                qkv=w('attn_qkv.weight')@h
                conv=silu(np.sum(hist[l].T*w('ssm_conv1d.weight')[:,:3],axis=1)+qkv*w('ssm_conv1d.weight')[:,3])
                hist[l][:-1]=hist[l][1:];hist[l][-1]=qkv
                q,k,v=np.split(conv,3)
                q=q/np.sqrt(np.sum(q*q)+1e-6)/np.sqrt(D);k=k/np.sqrt(np.sum(k*k)+1e-6)
                a=np.exp(np.logaddexp(0,(w('ssm_alpha.weight')@h)+w('ssm_dt.bias'))*w('ssm_a'))[0]
                b=sigmoid(w('ssm_beta.weight')@h)[0]
                states[l]*=a
                delta=(v-states[l]@k)*b
                states[l]+=np.outer(delta,k)
                out=states[l]@q
                out=rms(out,w('ssm_norm.weight'))*silu(w('attn_gate.weight')@h)
                y=w('ssm_out.weight')@out
            else:
                q,g=np.split(w('attn_q.weight')@h,2)
                q=rms(q,w('attn_q_norm.weight'));k=rms(w('attn_k.weight')@h,w('attn_k_norm.weight'))
                theta=pos*np.power(1e7,-2*np.arange(32)/64)
                for v in (q,k):
                    a=v[:32].copy();b=v[32:64].copy()
                    v[:32]=a*np.cos(theta)-b*np.sin(theta);v[32:64]=a*np.sin(theta)+b*np.cos(theta)
                kc.append(k);vc.append(w('attn_v.weight')@h)
                scores=np.array(kc)@q/16;scores=np.exp(scores-scores.max());scores/=scores.sum()
                y=w('attn_output.weight')@(scores@np.array(vc)*sigmoid(g))
            x+=y;h=rms(x,w('post_attention_norm.weight'))
            routing=w('ffn_gate_inp.weight')@h
            picked=np.argsort(-routing,kind='stable')[:K]
            probs=np.exp(routing[picked]-routing[picked].max());probs/=probs.sum()
            for e,p in zip(picked,probs):x+=p*(w('ffn_down_exps.weight')[e]@(silu(w('ffn_gate_exps.weight')[e]@h)*(w('ffn_up_exps.weight')[e]@h)))
            x+=sigmoid(w('ffn_gate_inp_shexp.weight')@h)*(w('ffn_down_shexp.weight')@(silu(w('ffn_gate_shexp.weight')@h)*(w('ffn_up_shexp.weight')@h)))
        logits.append(weights['output.weight']@rms(x,weights['output_norm.weight']))
    return logits

with tempfile.TemporaryDirectory() as td:
    td=Path(td);model=td/'tiny.gguf';writer=GGUFWriter(model,'qwen35moe')
    meta={'block_count':4,'embedding_length':N,'attention.head_count':1,'attention.head_count_kv':1,
          'attention.key_length':256,'attention.value_length':256,'expert_count':NE,'expert_used_count':K,
          'expert_feed_forward_length':FF,'expert_shared_feed_forward_length':FF,'ssm.conv_kernel':4,
          'ssm.state_size':D,'ssm.group_count':1,'ssm.time_step_rank':1,'ssm.inner_size':D,
          'full_attention_interval':4,'nextn_predict_layers':0,'rope.dimension_count':64}
    for k,v in meta.items():writer.add_uint32('qwen35moe.'+k,v)
    writer.add_array('tokenizer.ggml.tokens',[f't{i}' for i in range(V)])
    for name,w in weights.items():writer.add_tensor(name,w)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    exe=ROOT/'build-metal/strata-q36-metal';dump=td/'logits.f32'
    prompt=[1,7,13];forced=[9,3]
    subprocess.run([str(exe),'--native',str(model),'--tokens',','.join(map(str,prompt)),
                    '--n','3','--force',','.join(map(str,forced)),'--dump-full-logits',str(dump)],check=True,capture_output=True)
    got=np.fromfile(dump,np.float32).reshape(-1,V);want=np.array(cpu(prompt+forced)[len(prompt)-1:])
    relative=np.linalg.norm(got-want)/np.linalg.norm(want)
    assert np.isfinite(relative) and relative<1e-4,(relative,got[0,:5],want[0,:5])
    print(f'whole model, 3 teacher-forced logits: relative L2 {relative:.3e} PASS')
    # Two requests: the emitted token is not yet part of held recurrent state. Extension must reuse only the prompt.
    top=int(np.argmax(got[0]));lines=f'GEN 1 1,7,13\nGEN 1 1,7,13,{top}\nGEN 1 5000\n'
    result=subprocess.run([str(exe),'--native',str(model),'--serve'],input=lines,text=True,capture_output=True,check=True)
    output=result.stdout.splitlines();done=[x.split() for x in output if x.startswith('DONE ')]
    assert len(done)==2 and int(done[1][8])==3,output
    assert any(x.startswith('ERR token outside') for x in output),output
    emitted=[int(x.split()[1]) for x in output if x.startswith('T ')]
    assert emitted==[top,int(np.argmax(cpu(prompt+[top])[-1]))],emitted
    print('pipe requests, prefix reuse and invalid-token rejection PASS')

    import queue
    import threading
    proc=subprocess.Popen([str(exe),'--native',str(model),'--serve'],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    messages=queue.Queue()
    threading.Thread(target=lambda: [messages.put(x.strip()) for x in proc.stdout],daemon=True).start()
    try:
        assert messages.get(timeout=10).startswith('READY ')
        proc.stdin.write('GEN 64 '+','.join(['1']*1000)+'\n');proc.stdin.flush()
        assert messages.get(timeout=10).startswith('PP ')
        proc.stdin.write('STOP\n');proc.stdin.flush()
        while True:
            line=messages.get(timeout=10)
            if line.startswith('DONE '):
                assert line.split()[5]=='cancel',line
                break
        proc.stdin.write('GEN 1 1,7,13\n');proc.stdin.flush()
        emitted=[]
        while True:
            line=messages.get(timeout=10)
            if line.startswith('T '):emitted.append(int(line.split()[1]))
            if line.startswith('DONE '):break
        assert emitted==[top],emitted
        proc.stdin.close();proc.wait(timeout=10)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    print('prompt cancellation and subsequent request PASS')
