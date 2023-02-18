case $1 in

  'run')
      conda run --no-capture-output -n Error-Detection-at-Scale spark-submit --properties-file Configs/spark-defaults.conf ed_scale.py
    ;;

  'format')
    conda run --no-capture-output -n Error-Detection-at-Scale black .
    ;;

  'install')
    conda env create -f environment.yml
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
    
