case $1 in

  'run')
    spark-submit --master local[*] __main__.py
    ;;

  'format')
    black .
    ;;

  *)
    echo "Enter valid command"
    ;;
esac
    