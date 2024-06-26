package https

import (
	"go.uber.org/zap"
)

func listenRunnable(args []string) (response interface{}, err error) {

	var log *zap.Logger

	log.Info("https runnable")
	response = args

	return
}

func init() {

}
