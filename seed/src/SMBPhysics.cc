/// \file SMBPhysics.cc
/// \brief Implementation of the SMBPhysics class.

#include "SMBPhysics.hh"

#include <iomanip>   
#include <CLHEP/Units/SystemOfUnits.h>

#include "globals.hh"
#include "G4ios.hh"

#include "G4DecayPhysics.hh"
// #include "G4EmStandardPhysics.hh"  // standard for hep
#include "G4EmStandardPhysics_option1.hh"  // for hep, fast not precise
// #include "G4EmStandardPhysics_option4.hh"  // most accurate EM models
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4RadioactiveDecayPhysics.hh"

// #include "QGSP_BERT.hh"
// #include "G4HadronPhysicsQGSP_BERT.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"

#include "G4UnitsTable.hh"
#include "G4EmParameters.hh"

#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SMBPhysics::SMBPhysics() 
: G4VModularPhysicsList(){
  SetVerboseLevel(1);

  // EM Physics
  RegisterPhysics( new G4EmStandardPhysics_option1() );  // change accordingly
  // making sure cuts can be set
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetApplyCuts(true);

  // Synchroton Radiation & GN Physics
  RegisterPhysics( new G4EmExtraPhysics() );

  // Decays
  RegisterPhysics( new G4DecayPhysics() );

   // Hadron Elastic scattering
  RegisterPhysics( new G4HadronElasticPhysics() );

  // Hadron Physics
  // RegisterPhysics( new G4HadronPhysicsQGSP_BERT());
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT());

  // Stopping Physics
  RegisterPhysics( new G4StoppingPhysics() );

  // Ion Physics
  RegisterPhysics( new G4IonPhysics());
  
  // Neutron tracking cut
  RegisterPhysics( new G4NeutronTrackingCut());

  // Radioactive decay

  RegisterPhysics(new G4RadioactiveDecayPhysics());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SMBPhysics::~SMBPhysics()
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SMBPhysics::SetCuts()
{
  // fixe lower limit for cut

  G4double lowlimit = 0.25 * keV;

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(
      lowlimit, 100. * TeV);

  // call base class method to set cuts which default value can be

  // modified via /run/setCut/* commands

  G4VUserPhysicsList::SetCuts();

  DumpCutValuesTable();
}  
