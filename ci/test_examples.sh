for example in $(find ./examples/ -iname *.py); do
  CI_MODE=1 python $example
done
