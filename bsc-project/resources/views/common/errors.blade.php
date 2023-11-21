@if (count($errors) > 0)
    @foreach ($errors->all() as $error)
        <div class="uk-alert uk-alert-danger" data-uk-alert="">
            <a href="#" class="uk-alert-close uk-close"></a>
            {{ $error }}
        </div>
    @endforeach
@endif

