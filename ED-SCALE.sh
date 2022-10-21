case $1 in

  'run')
      spark-submit --master local[*] --driver-memory 50G __main__.py
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
    
