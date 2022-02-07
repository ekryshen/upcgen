//////////////////////////////////////////////////////////////////////////
// Copyright (C) 2021, Nazar Burmasov, Evgeny Kryshen
//
// E-mail of the corresponding author: nazar.burmasov@cern.ch
//
// This file is a part of Upcgen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//////////////////////////////////////////////////////////////////////////

#include "UpcTwoPhotonLbyL.h"

#include "TFile.h"

UpcTwoPhotonLbyL::UpcTwoPhotonLbyL()
{
  mPart = 0.0;
  partPDG = 22;
  isCharged = false;

  // load cross sections from files
  auto* f_m = new TFile("../cross_sections/lbyl/cross_section_m.root");
  hCrossSectionM = (TH1D*)f_m->Get("hCrossSectionM");
  hCrossSectionM->SetDirectory(nullptr);
  f_m->Close();

  auto* f_zm = new TFile("../cross_sections/lbyl/cross_section_zm.root");
  hCrossSectionZM = (TH2D*)f_zm->Get("hCrossSectionZM");
  hCrossSectionZM->SetDirectory(nullptr);
  f_zm->Close();
}

double UpcTwoPhotonLbyL::calcCrossSectionM(double m)
{
  int binM = hCrossSectionM->GetXaxis()->FindBin(m);
  double cs = hCrossSectionM->GetBinContent(binM);
  return cs;
  // [nb]
}

double UpcTwoPhotonLbyL::calcCrossSectionZM(double z, double m)
{
  int binZ = hCrossSectionZM->GetXaxis()->FindBin(z);
  int binM = hCrossSectionZM->GetYaxis()->FindBin(m);
  double cs = hCrossSectionZM->GetBinContent(binZ, binM);
  return cs;
}
