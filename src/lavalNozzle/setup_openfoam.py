import os

def generate_field_file(filepath, class_name, object_name, dims, internal, patch_map):
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {dims};

internalField   {internal};

boundaryField
{{
"""
    for patch, settings in patch_map.items():
        content += f"    {patch}\n    {{\n"
        for key, value in settings.items():
            content += f"        {key} {value};\n"
        content += "    }\n"
        
    content += "}\n\n// ************************************************************************* //\n"
    
    with open(filepath, 'w') as f:
        f.write(content)

def setup_case(case_dir):
    # S'assurer que le répertoire system existe
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)

    # Champ U
    u_map = {
        "inlet": {"type": "pressureInletOutletVelocity", "value": "uniform (10 0 0)"},
        "outlet": {"type": "pressureInletOutletVelocity", "value": "uniform (10 0 0)"},
        "nozzle": {"type": "noSlip"},
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/U"), "volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (10 0 0)", u_map)
    
    # Champ p
    p_map = {
        "inlet": {"type": "totalPressure", "p0": "uniform 1000000", "value": "uniform 1000000", "gamma": "1.4", "psi": "thermo:psi", "U": "U"},
        "outlet": {"type": "waveTransmissive", "field": "p", "psi": "thermo:psi", "gamma": "1.4", "fieldInf": "35000", "lInf": "0.1", "value": "uniform 35000"},
        "nozzle": {"type": "zeroGradient"},
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/p"), "volScalarField", "p", "[1 -1 -2 0 0 0 0]", "uniform 35000", p_map)

    # Champ T
    T_map = {
        "inlet": {"type": "totalTemperature", "T0": "uniform 3000", "value": "uniform 3000", "gamma": "1.4", "psi": "thermo:psi", "U": "U"},
        "outlet": {"type": "zeroGradient" },
        "nozzle": {"type": "zeroGradient"},
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/T"), "volScalarField", "T", "[0 0 0 1 0 0 0]", "uniform 1000", T_map)
    
    # Champ k
    k_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "nozzle": {"type": "kLowReWallFunction", "value": "uniform 1e-10"},
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/k"), "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 0.375", k_map)

    # Champ omega
    omega_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "nozzle": {"type": "omegaWallFunction", "value": "uniform 1e-10"},
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/omega"), "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 5000", omega_map)

    # Champ nut
    nut_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "symmetry": {"type": "symmetryPlane"},
        "nozzle": {"type": "nutLowReWallFunction", "value": "uniform 0"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/nut"), "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", nut_map)

    # Champ alphat
    alphat_map = {
        "inlet": {"type": "calculated", "value": "uniform 0"},
        "outlet": {"type": "calculated", "value": "uniform 0"},
        "nozzle": {"type": "compressible::alphatWallFunction", "value": "uniform 0"}, 
        "symmetry": {"type": "symmetryPlane"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/alphat"), "volScalarField", "alphat", "[1 -1 -1 0 0 0 0]", "uniform 0", alphat_map)

    # Mettre à jour les types de frontières
    boundary_path = os.path.join(case_dir, "constant/polyMesh/boundary")
    if os.path.exists(boundary_path):
        with open(boundary_path, 'r') as f:
            content = f.read()
        import re
        content = content.replace("physicalType    patch;", "")
        content = re.sub(r'(front\s*\{\s*type\s*)patch', r'\1empty', content)
        content = re.sub(r'(back\s*\{\s*type\s*)patch', r'\1empty', content)
        content = re.sub(r'(nozzle\s*\{\s*type\s*)patch', r'\1wall', content)
        content = re.sub(r'(symmetry\s*\{\s*(?:[^{}]*?\s+)?type\s+)patch', r'\1symmetryPlane', content)
        content = re.sub(r'(inlet\s*\{\s*type\s*)patch', r'\1inlet', content)
        content = re.sub(r'(outlet\s*\{\s*type\s*)patch', r'\1outlet', content)
        with open(boundary_path, 'w') as f:
            f.write(content)

    # TurbulenceProperties
    with open(os.path.join(case_dir, "constant/turbulenceProperties"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType          RAS;
RAS
{
    RASModel            kOmegaSST;
    turbulence          on;
    printCoeffs         on;
}
""")

    # ThermophysicalProperties
    with open(os.path.join(case_dir, "constant/thermophysicalProperties"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      thermophysicalProperties;
}

thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie
    {
        molWeight       28.96;
    }
    thermodynamics
    {
        Cp              1004.5;
        Hf              0;
    }
    transport
    {
        As              1.458e-06;
        Ts              110.4;
    }
}
""")

    # fvSchemes
    with open(os.path.join(case_dir, "system/fvSchemes"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss limitedLinear 1;
    div(phi,e)      Gauss limitedLinear 1;
    div(phi,K)      Gauss limitedLinear 1;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,omega)  Gauss limitedLinear 1;
    div(phid,p)     Gauss limitedLinear 1;
    div(tauMC)      Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
interpolationSchemes
{
    default         linear;
    reconstruct(rho) vanLeer;
    reconstruct(U)  vanLeerV;
    reconstruct(T)  vanLeer;
}
snGradSchemes   { default corrected; }
wallDist
{
    method meshWave;
}
""")

    # fvSolution
    with open(os.path.join(case_dir, "system/fvSolution"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    "rho.*"
    {
        solver          diagonal;
    }

    "(U|e|k|omega|up|BCp).*"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
}
""")

    # fvOptions
    with open(os.path.join(case_dir, "system/fvOptions"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvOptions;
}
limitTemp
{
    type            limitTemperature;
    active          yes;
    selectionMode   all;
    min             200;  
    max             4000;
}
limitU
{
    type            limitVelocity;
    active          yes;
    selectionMode   all;
    max             3000;
}
limitNut
{
    type            limitTurbulenceViscosity;
    active          yes;
    selectionMode   all;
    max             1.0;
}
""")


    # controlDict
    with open(os.path.join(case_dir, "system/controlDict"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     rhoCentralFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.003;          
deltaT          1e-7;          
writeControl    adjustableRunTime;
writeInterval   0.0002;         
purgeWrite      1;
writeFormat     ascii;
writePrecision  6;
adjustTimeStep  yes;          
maxCo           0.3;           

functions
{
    massFlowInlet
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    timeStep;
        writeFields     false;
        log             true;
        operation       sum;
        regionType      patch;
        name            inlet;
        fields          (phi);
    }
    massFlowOutlet
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    timeStep;
        writeFields     false;
        log             true;
        operation       sum;
        regionType      patch;
        name            outlet;
        fields          (phi);
    }
}
""")
    
    # decomposeParDict
    with open(os.path.join(case_dir, "system/decomposeParDict"), 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2512                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains 4;

method          scotch;
""")

if __name__ == "__main__":
    case = "simulations/nozzle/nozzle_test"
    setup_case(case)
    print("Setup - ensure parallel config complete.")
