//---------------------------------------------------------------------------
//
// ClassName:   SMBPhysics
//
// Author: 2021 S. Meighen-Berger
//
// Modified:
//
//----------------------------------------------------------------------------
//
#ifndef SMBPhysics_h
#define SMBPhysics_h 1

#include "globals.hh"
#include "G4VModularPhysicsList.hh"

/// Modular physics list
///
/// It includes the folowing physics builders
/// - G4EmStandardPhysics
/// - G4EmExtraPhysics
/// - G4DecayPhysics
/// - G4HadronElasticPhysics
/// - G4HadronPhysicsQGSP_BERT
/// - G4StoppingPhysics
/// - G4IonPhysics
/// - G4NeutronTrackingCut


class SMBPhysics: public G4VModularPhysicsList
{
public:
  SMBPhysics();
  virtual ~SMBPhysics();

  virtual void SetCuts();
};


#endif