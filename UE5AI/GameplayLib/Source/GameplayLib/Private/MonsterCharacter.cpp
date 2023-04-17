// MonsterCharacter.cpp

#include "MonsterCharacter.h"
#include "AIController.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "BehaviorTree/BlackboardComponent.h"

AMonsterCharacter::AMonsterCharacter()
{
    // Set default values
    Health = 100.0f;
	SightRange = 2000.0f;
    AttackRange = 200.0f;
    MovementState = EMonsterMovementState::Idle;
}

void AMonsterCharacter::BeginPlay()
{
    Super::BeginPlay();
}

float AMonsterCharacter::TakeDamage(float DamageAmount, const FDamageEvent& DamageEvent, AController* EventInstigator, AActor* DamageCauser)
{
    float ActualDamage = Super::TakeDamage(DamageAmount, DamageEvent, EventInstigator, DamageCauser);
    Health -= ActualDamage;

    if (Health <= 0)
    {
        MovementState = EMonsterMovementState::Corpse;
        GetCharacterMovement()->DisableMovement();
        NotifyDeath();
    }

    return ActualDamage;
}

void AMonsterCharacter::NotifyDeath()
{
    AAIController* AIController = Cast<AAIController>(GetController());
    if (AIController)
    {
        UBlackboardComponent* BlackboardComponent = AIController->GetBlackboardComponent();
        if (BlackboardComponent)
        {
            BlackboardComponent->SetValueAsBool(TEXT("IsDead"), true);
        }
    }
}

void AMonsterCharacter::Attack(AActor* AttackTarget)
{
    UE_LOG(LogTemp, Warning, TEXT("Monster::Attack"));
    OnAttack(AttackTarget);
}

void AMonsterCharacter::OnAttack_Implementation(AActor* Actor)
{
    
}
