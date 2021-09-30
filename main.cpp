//
// Created by nburmaso on 6/25/21.
//

#include "LepGenerator.h"
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Appenders/ColorConsoleAppender.h>

int main(int argc, char** argv)
{
  // logging levels:
  //  0 -> no debug logs
  //  1 -> basic debug logs
  //  2 -> verbose logs with intermediate calculation results
  PLOG_FATAL_IF(argc < 2) << "Wrong number of parameters! Usage: generate debug_level";

  // initialize logger
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::verbose, &consoleAppender);

  // input
  int debugLevel = std::stoi(argv[1]);
  PLOG_INFO << "Debug level: " << debugLevel;

  PLOG_INFO << "Initializing the generator...";
  auto* lepGenerator = new LepGenerator();
  lepGenerator->setDebugLevel(debugLevel);
  lepGenerator->initGeneratorFromFile();

  PLOG_INFO << "Starting generation process...";
  lepGenerator->generateEvents();

  PLOG_INFO << "Event generation is finished!";

  return 0;
}