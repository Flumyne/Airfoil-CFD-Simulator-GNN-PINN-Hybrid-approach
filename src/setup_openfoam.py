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
        "inlet": {"type": "freestreamVelocity", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestreamVelocity", "freestreamValue": "$internalField"},
        "airfoil": {"type": "noSlip"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/U"), "volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (25 0 0)", u_map)
    
    # Champ p
    p_map = {
        "inlet": {"type": "freestreamPressure", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestreamPressure", "freestreamValue": "$internalField"},
        "airfoil": {"type": "zeroGradient"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/p"), "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", p_map)
    
    # Champ k
    k_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "airfoil": {"type": "kLowReWallFunction", "value": "uniform 1e-10"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/k"), "volScalarField", "k", "[0 2 -2 0 0 0 0]", "uniform 0.1", k_map)

    # Champ omega
    omega_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "airfoil": {"type": "omegaWallFunction", "value": "uniform 1e-10"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/omega"), "volScalarField", "omega", "[0 0 -1 0 0 0 0]", "uniform 10", omega_map)

    # Champ nut
    nut_map = {
        "inlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "outlet": {"type": "freestream", "freestreamValue": "$internalField"},
        "airfoil": {"type": "nutkWallFunction", "value": "uniform 0"},
        "front": {"type": "empty"},
        "back": {"type": "empty"}
    }
    generate_field_file(os.path.join(case_dir, "0/nut"), "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", nut_map)

    # Mettre à jour les types de frontières
    boundary_path = os.path.join(case_dir, "constant/polyMesh/boundary")
    if os.path.exists(boundary_path):
        with open(boundary_path, 'r') as f:
            content = f.read()
        import re
        content = content.replace("physicalType    patch;", "")
        content = re.sub(r'(front\s*\{\s*type\s*)patch', r'\1empty', content)
        content = re.sub(r'(back\s*\{\s*type\s*)patch', r'\1empty', content)
        content = re.sub(r'(airfoil\s*\{\s*type\s*)patch', r'\1wall', content)
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
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
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
    p { solver GAMG; tolerance 1e-07; relTol 0.05; smoother GaussSeidel; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-08; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl { p 1e-5; U 1e-5; k 1e-5; omega 1e-5; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
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
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   200;
purgeWrite      2;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
functions
{
    forceCoeffs1
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0.25 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         25;
        lRef            1;
        Aref            0.1;
    }
    yPlus
    {
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
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
    case = "simulations/naca2412_test"
    setup_case(case)
    print("Setup - ensure parallel config complete.")
