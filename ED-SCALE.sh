case $1 in

  'run')
    if [ -z "$2" ] 
    then
      spark-submit --master local[*] __main__.py
    else
      spark-submit --master local[$2] __main__.py
    fi
    ;;

  'format')
    black .
    ;;

  'install')
    conda env create -f edatscale-env.yml
    ;;

  'activate')
    conda activate Error-Detection-at-Scale
    ;;

  *)
    echo "Enter valid command"
    ;;
esac
    
