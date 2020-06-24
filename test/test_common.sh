# Common functions for shell tests. "Ported" from iharm3d

run_harm () {
    ../../run.sh -i ../${1}.par \
        parthenon/mesh/nx1=${2} parthenon/mesh/nx1=${3} parthenon/mesh/nx1=${4} \
        coordinates/transform=${5}
}