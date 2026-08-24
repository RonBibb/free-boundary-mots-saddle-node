#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.v1_solver import continue_v1

def main():
    p=argparse.ArgumentParser();p.add_argument("--amplitudes",type=float,nargs="+",required=True);p.add_argument("--nz",type=int,default=25);p.add_argument("--nr",type=int,default=37);p.add_argument("--output",type=Path,default=Path("results/v1_energy_continuation.json"));a=p.parse_args()
    result=continue_v1(a.amplitudes,nz=a.nz,nr=a.nr,tolerance=1e-9,iterations=100)
    result["grid"]=[a.nz,a.nr];result["continuation_coordinate_reported"]="energy_dimensionless"
    a.output.parent.mkdir(exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
