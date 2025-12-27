# Julia AMD GPU test 

### Install the latest Julia 

```bash
curl -fsSL https://install.julialang.org | sh
```

Install the AMDGPU package

```bash
julia --project=. -e 'import Pkg; Pkg.add("AMDGPU")'
```

or globally 

```bash
julia  -e 'import Pkg; Pkg.add("AMDGPU")'
```


### Install latest rcom ( I have a 9060)

I was able to have this pass by udating from rcom 7.0 to 7.1

To install the latests rcom, follow the instructions [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)