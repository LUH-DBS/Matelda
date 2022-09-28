case $1 in

  'run')
    spark-submit --master local[*] end-to-end-eds.py
    ;;

  'format')
    black .
    ;;

  *)
    echo "Enter valid command"
    ;;
esac
    