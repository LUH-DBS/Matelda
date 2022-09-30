case $1 in

  'run')
    conda activate Error-Detection-at-Scale
    spark-submit --master local[*] __main__.py
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
    
