#!/bin/bash
find . -type f -name '*.pdf' -print0 |
  while IFS= read -r -d '' file
    do pdftoppm "${file}" "${file%.*}" -png -rx 300 -ry 300
  done
    
