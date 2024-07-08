for example in $(find ./examples/ -iname *.py); do
  CI_MODE=True python $example
done
