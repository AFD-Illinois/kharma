#!/bin/bash
# Cleans all temporary/gitignore files from tests

TEST_DIR=$(dirname "$(readlink -f "$0")")
rm -rf ${TEST_DIR}/*/*.{phdf,xdmf,rhdf,hst,txt,png} ${TEST_DIR}/tilt_init/mks ${TEST_DIR}/*/frames_* ${TEST_DIR}/*/kharma_parsed_parameters*
