case $1 in

  'run')
      spark-submit --properties-file Configs/spark-defaults.conf ed_scale.py
    ;;

  'format')
    black .
    ;;

  'install')
    conda env create -f environment.yml
    ;;

  'activate')
    conda activate Error-Detection-at-Scale
    ;;

  'remove')
    conda env remove -n Error-Detection-at-Scale
    ;;
  
  'clearraha')
    find ./ -name "raha-baran-results-*" -print0 | xargs -0 rm -r 
    ;;
  *)
    echo "Enter valid command"
    ;;
esac
    
